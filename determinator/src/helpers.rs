use core::f64;
use std::{
    borrow::BorrowMut,
    env,
    num::FpCategory,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use diesel::{Connection, PgConnection};
use dotenvy::dotenv;
use feature_database::{
    elevationdb::geotransform::get_world_coordinates, keypointdb::KeypointDatabase,
};
use feature_extraction::{
    akaze_keypoint_descriptor_extraction_def, get_knn_matches, get_points_from_matches,
};
use homographier::homographier::{Cmat, ImgObjCorrespondence, MatError, PNPRANSACSolution};
use opencv::{
    core::{
        hconcat2, KeyPoint, KeyPointTraitConst, Mat, MatExprTraitConst, MatTrait, MatTraitConst,
        MatTraitConstManual, Point2d, Point2f, Point3_, Point3d, Point3f, Point_, Size2i, Vec4d,
        Vector,
    },
    calib3d,
    imgcodecs::{IMREAD_COLOR, IMREAD_GRAYSCALE},
};
use rgb::alt::BGR8;

use nalgebra::{Dyn, Matrix, OMatrix, SMatrix, Vector3, Vector4, U1, U3, U4};

use crate::{Args, CameraIntrinsic};

type DbType = Arc<Mutex<diesel::PgConnection>>;

#[derive(Debug, Clone)]
pub enum Coordinates3d {
    Ellipsoidal { lat: f32, lon: f32, height: f32 },
    Cartesian { x: f64, y: f64, z: f64 },
}

#[derive(Debug)]
pub struct Cartesian3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone)]
pub enum Coordinates2d {
    Ellipsoidal { lat: f32, lon: f32 },
    Cartesian { x: f64, y: f64 },
}

impl From<Coordinates3d> for Coordinates2d {
    fn from(value: Coordinates3d) -> Self {
        match value {
            Coordinates3d::Ellipsoidal {
                lat,
                lon,
                height: _,
            } => Coordinates2d::Ellipsoidal { lat, lon },
            Coordinates3d::Cartesian { x, y, z: _ } => Coordinates2d::Cartesian { x, y },
        }
    }
}

pub struct ReadAndExtractKpResult(pub Cmat<BGR8>, pub Vector<KeyPoint>, pub Cmat<u8>);

#[derive(Debug)]
pub struct EulerAngles {
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
}

impl EulerAngles {
    pub fn to_deg(&self) -> EulerAngles {
        use std::f64::consts::PI;
        EulerAngles {
            roll: self.roll * 180.0 / PI,
            pitch: self.pitch * 180.0 / PI,
            yaw: self.yaw * 180.0 / PI,
        }
    }
}

pub fn read_and_extract_kp(im_path: &Path) -> (ReadAndExtractKpResult, (i64, i64)) {
    if !im_path.is_file() {
        panic!("{:?} Provided image path does not point to a file", im_path);
    }

    assert!(
        matches!(im_path.extension().expect("Image has no file extension").to_str(), Some(val) if ["png", "jpg", "jpeg", "tif", "tiff"].contains(&val.to_lowercase().as_str())),
        "input image of file format {:?} is unsupported",
        im_path.extension()
    );
    let path = im_path
        .to_str()
        .expect("provided image path is not valid unicode");

    // it is assumed that input images will not contain an alpha channel
    let image = Cmat::<BGR8>::imread_checked(path, IMREAD_COLOR).expect("Failed to read image");

    let resolution = image.mat.size().expect("Could not read image dimensions");

    let extracted = akaze_keypoint_descriptor_extraction_def(&image.mat, None)
        .expect("AKAZE keypoint extraction failed");

    (
        ReadAndExtractKpResult(
            image,
            extracted.keypoints,
            Cmat::<u8>::new(extracted.descriptors).expect("Matrix construction should not fail"),
        ),
        (resolution.width as i64, resolution.height as i64),
    )
}

pub fn get_camera_matrix(cam_intrins: CameraIntrinsic) -> Result<Cmat<f64>, ()> {
    // [[f_x,s  ,c_x],
    //  [0.0,f_y,c_y],
    //  [0.0,0.0,1.0]]
    let mat: Cmat<f64> = match cam_intrins {
        CameraIntrinsic::File { path } => parse_into_matrix(path)?,
        CameraIntrinsic::Manual {
            focal_len_x,
            focal_len_y,
            skew,
            offset_x,
            offset_y,
        } => {
            let arr: [[f64; 3]; 3] = [
                [focal_len_x, skew, offset_x],
                [0.0, focal_len_y, offset_y],
                [0.0, 0.0, 1.0],
            ];
            Cmat::from_2d_slice(&arr).map_err(|_err| ())?
        }
    };
    Ok(mat)
}

fn parse_into_matrix(path: String) -> Result<Cmat<f64>, ()> {
    let path = PathBuf::from(path);
    match path.extension().ok_or(())?.to_str().ok_or(())? {
        "json" => {
            unimplemented!("har ikke opsat parser :)")
        }
        _ => Err(()),
    }
}

/// Finds Point correspondences between keypoints found in the input image and reference image
/// # Parameters
/// * args: command line arguments, used to read the input image and to determine what to consider as reference image
/// * image: the input image and associated keypoints + descriptors
/// # Notes
/// If the program is set to demo mode, the input image is assumed to be a 7x7 chessboard pattern. Otherwise knn matching is attempted
/// # Panics
/// Will panic if demo mode is enabled and the input image is not a 7x7 chessboard pattern.
/// Will panic on any database error
pub fn img_obj_corres(args: &Args, image: ReadAndExtractKpResult) -> Vec<ImgObjCorrespondence> {
    let (ref_keypoints, ref_descriptors) = ref_keypoints(args);
    // dbg!((&ref_descriptors.clone().unwrap().len(),&ref_keypoints.len()));
    match ref_descriptors {
        Some(val) => matching_with_descriptors(
            &image,
            &Cmat::<u8>::from_2d_slice(&val).expect("Failed to convert keypoints to matrix"),
            Vector::from_iter(ref_keypoints),
        )
        .expect("keypoint matching failed"),
        None => {
            //TODO: move to dedicated function
            let mut img_points: Vector<Point2f> = Vector::new();
            opencv::calib3d::find_chessboard_corners_def(
                &Cmat::<u8>::imread_checked(&args.img_path.to_string_lossy(), IMREAD_GRAYSCALE)
                    .expect("failed to read image")
                    .mat,
                Size2i::new(9, 9),
                &mut img_points,
            )
            .expect("failed to find chessboard corners")
            .then_some(())
            .expect("failed to find chessboard corners");
            assert_eq!(ref_keypoints.len(), img_points.len());
            let corres = ref_keypoints
                .into_iter()
                .zip(img_points)
                .map(|(o, i)| {
                    ImgObjCorrespondence::new(
                        Point3d::from_point(o.pt().to().expect("f32 cast to f64 should not fail")),
                        i.to().expect("f32 cast to f64 should not fail"),
                    )
                })
                .collect::<Vec<_>>();
            corres
        }
    }
}

//TODO: DENNE FUNKTION ER CURSED
fn matching_with_descriptors(
    img: &ReadAndExtractKpResult,
    ref_desc: &Cmat<u8>,
    ref_kp: Vector<KeyPoint>,
) -> Result<Vec<ImgObjCorrespondence>, opencv::Error> {
    let matches = get_knn_matches(&img.2.mat, &ref_desc.mat, 2, 0.8)?;
    // dbg!(&ref_desc.mat.size());
    // dbg!(&img.2.mat.size());
    // dbg!(&matches);
    // dbg!(&img.1);
    // dbg!(&ref_kp);
    let (img_points, obj_points_2d) = get_points_from_matches(&img.1, &ref_kp, &matches)?;
    assert_eq!(img_points.len(), obj_points_2d.len());
    // dbg!(&img_points);
    // dbg!(&obj_points_2d);
    // map object points to real world coordinates
    let obj_points = get_3d_world_coord_from_2d_point(
        obj_points_2d
            .into_iter()
            .inspect(|f| {
                dbg!(f);
            })
            .map(|f| Point2d::new(f.x as f64, f.y as f64))
            .collect(),
        Arc::new(Mutex::new(
            PgConnection::establish(
                env::var("DATABASE_URL")
                    .expect("failed to read DATABASE_URL")
                    .as_str(),
            )
            .expect("failed to establish connection"),
        )),
    );
    // dbg!(&obj_points);
    Ok(point_pair_to_correspondence(
        img_points,
        obj_points
            .into_iter()
            .map(|f| Point3f::new(f.x as f32, f.y as f32, f.z as f32))
            .collect::<Vec<_>>(),
    ))
}

fn point_pair_to_correspondence(
    img_points: Vector<Point_<f32>>,
    obj_points: Vec<opencv::core::Point3_<f32>>,
) -> Vec<ImgObjCorrespondence> {
    img_points
        .into_iter()
        .zip(obj_points)
        .map(|f| ImgObjCorrespondence::new(point3f_to3d(f.1), point2f_to2d(f.0)))
        .collect()
}

pub fn ref_keypoints(args: &Args) -> (Vec<KeyPoint>, Option<Vec<Vec<u8>>>) {
    match args.demo {
        true => {
            // TIHI @Murmeldyret, here be no side effects
            let points: Result<Vec<KeyPoint>, _> = (-4..=4)
                .map(|f| (f, (-4..=4)))
                .flat_map(|row| {
                    row.1.map(move |col| {
                        KeyPoint::new_coords_def(row.0 as f32 * 1.9, col as f32 * 1.9, 1.0)
                    })
                })
                .collect();
            (points.expect("Failed to create keypoints"), None)
        }
        false => {
            let (keypoints_from_db, descriptors) = keypoints_from_db(
                &env::var("DATABASE_URL").expect("failed to read DATABASE_URL"),
                args,
            );
            (keypoints_from_db, Some(descriptors))
        }
    }
}

fn keypoints_from_db(conn_url: &str, arg: &Args) -> (Vec<KeyPoint>, Vec<Vec<u8>>) {
    let conn: DbType = Arc::new(Mutex::new(
        Connection::establish(conn_url).expect("Failed to connect to database"),
    ));

    // retrieve keypoints from mosaic image
    let ref_keypoints = feature_database::keypointdb::Keypoint::read_keypoints_from_lod(
        &mut conn.lock().expect("Mutex poisoning"),
        arg.lod,
    )
    .expect("Failed to query database");
    // Map keypoints to opencv compatible type
    let (ref_keypoints, ref_descriptors) = db_kp_to_opencv_kp(ref_keypoints);

    (ref_keypoints, ref_descriptors)
}

fn db_kp_to_opencv_kp(
    ref_keypoints: Vec<feature_database::models::Keypoint>,
) -> (Vec<KeyPoint>, Vec<Vec<u8>>) {
    let (ref_keypoints, ref_descriptors): (Vec<_>, Vec<_>) = ref_keypoints
        .into_iter()
        .map(|f| {
            (
                opencv::core::KeyPoint::new_point(
                    Point2f::new(f.x_coord, f.y_coord),
                    f.size,
                    f.angle,
                    f.response,
                    f.octave,
                    f.class_id,
                )
                .expect("error in converting db keypoint to opencv keypoint"),
                f.descriptor,
            )
        })
        .unzip();
    (ref_keypoints, ref_descriptors)
}

// TODO: kan godt være topo parameter skal ændres til en anden type
/// Maps a 2d keypoint to a 3d object point, for use in creating image-object point correspondences
/// # Parameters
/// * points: a vector of keypoints from a reference image
/// * db: connection to database
/// # Returns
/// a vector of object points in real world coordinates (/*TODO: cartesian or ellipsoidal coordinates? */)
pub fn get_3d_world_coord_from_2d_point(points: Vec<Point2d>, db: DbType) -> Vec<Point3d> {
    points
        .into_iter()
        .map(|p| {
            let world_coord: (f64, f64, f64) =
                get_world_coordinates(db.lock().expect("mutex poisoning").borrow_mut(), p.x, p.y)
                    .expect("failed to retrieve world coordinates");
            Point3d::new(world_coord.0, world_coord.1, world_coord.2)
        })
        .collect::<Vec<_>>()
}

pub fn point2f_to2d(p: Point2f) -> Point2d {
    Point2d::new(p.x as f64, p.y as f64)
}

pub fn point3f_to3d(p: Point3f) -> Point3d {
    Point3d::new(p.x as f64, p.y as f64, p.z as f64)
}
// https://docs.opencv.org/4.x/dc/d2c/tutorial_real_time_pose.html
/// Finds where a given point would appear on the image
/// # Returns
/// a 2d homogenous point (z=1)
pub fn project_obj_point(
    obj_point: Point3d,
    solution: PNPRANSACSolution,
    cam_mat: Cmat<f64>,
) -> Point3d {
    // let tvec = [first,second,third];
    let mut rt_mat = Cmat::<f64>::zeros(4, 4).expect("Matrix initialization should not fail");
    hconcat2(&solution.rvec.mat, &solution.tvec.mat, &mut rt_mat.mat)
        .expect("Matrix operation should not fail");
    let rt_mat = rt_mat;

    // homogenous object point
    let obj_point_hom = Vec4d::new(obj_point.x, obj_point.y, obj_point.z, 1f64).to_vec();

    // dbg!(&obj_point_hom);
    // println!("{:?}\n{:?}\n{:?}\n",rt_mat.mat.at_row::<f64>(0).unwrap(),rt_mat.mat.at_row::<f64>(1).unwrap(),rt_mat.mat.at_row::<f64>(2).unwrap());

    let temp = (cam_mat.mat * rt_mat.mat)
        .into_result()
        .expect("matrix expression should not failed")
        .to_mat()
        .expect("matrix construction should not fail");

    assert_eq!(temp.rows(),3,"Matrix multiplication between calibration matrix (A) and rotation+translation matix (Rt) should yield a 3X4 matrix");
    assert_eq!(temp.cols(),4,"Matrix multiplication between calibration matrix (A) and rotation+translation matix (Rt) should yield a 3X4 matrix");

    // println!("{:?}\n{:?}\n{:?}\n",temp.at_row::<f64>(0).unwrap(),temp.at_row::<f64>(1).unwrap(),temp.at_row::<f64>(2).unwrap());
    let mut result: [f64; 3] = [0f64; 3];
    // this is because opencv does not allow matrix vector product for some reason :(
    for (i, e) in result.iter_mut().enumerate().take(temp.rows() as usize) {
        *e = temp
            .at_row::<f64>(i as i32)
            .expect("row index should be within range")
            .iter()
            .enumerate()
            .map(|(i, e)| {
                e * obj_point_hom
                    .get(i)
                    .expect("index should not be out of range")
            })
            // .inspect(|f|{dbg!(f);})
            .reduce(|acc, elem| acc + elem)
            .expect("Reduce operation should yield a value");
    }
    Point3d::new(result[0] / result[2], result[1] / result[2], 1f64)
    // let vector = temp.iter::<f64>().unwrap().map(|(p,e)|e*obj_point_hom.get(p.y as usize).unwrap()).collect::<Vec<f64>>();
    // let rhs = dbg!(temp.dot(&obj_point_hom));
    // dbg!(&rhs);
    // let _ = temp.iter::<f64>().unwrap().inspect(|f|println!("({},{}) = {}",f.0.x,f.0.y,f.1)).collect::<Vec<_>>();
    // let rhs = temp.elem_mul(obj_point_hom).into_result().unwrap().to_mat().unwrap();
}

/// implementation of equation found here: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
/// Maps a coordinate expressed in the world (global) frame to a coordinate expressed in the camera (local) frame
/// # Note
/// If the function returns (0,0,0), that means that the given object point is the world coordinate of the camera
///
pub fn world_frame_to_camera_frame(obj_point: Point3d, solution: &PNPRANSACSolution) -> Point3d {
    let obj_point_hom = Vec4d::new(obj_point.x, obj_point.y, obj_point.z, 1f64);

    let mut rt_mat = Cmat::<f64>::zeros(4, 4).expect("matrix initialization should not fail");
    hconcat2(&solution.rvec.mat, &solution.tvec.mat, &mut rt_mat.mat)
        .expect("Matrix operation should not fail");

    // add row with values [0,0,0,1]
    rt_mat.mat.resize(4).expect("matrix resize should not fail");
    *rt_mat
        .mat
        .at_2d_mut(3, 3)
        .expect("index should be within range") = 1f64;
    let rt_mat = rt_mat.mat;

    let mut result: [f64; 4] = [0f64; 4];
    for i in 0..rt_mat.rows() {
        result[i as usize] = rt_mat
            .at_row::<f64>(i)
            .expect("should be within bounds")
            .iter()
            .enumerate()
            .map(|(i, e)| {
                e * obj_point_hom
                    .get(i)
                    .expect("index should not be out of range")
            })
            // .inspect(|f|{dbg!(f);})
            .reduce(|acc, elem| acc + elem)
            .expect("Reduce operation should yield a value");
    }
    Point3d::new(result[0], result[1], result[2])
}

/// Ensures that command line arguments are valid, panics otherwise
pub fn validate_args(args: &Args) {
    if let Some(coeffs) = &args.dist_coeff {
        assert!(
            matches!(coeffs.len(), 4 | 5 | 8 | 12 | 14),
            "Distortion coefficient length does not have required length of 4|5|8|12|14, found {}",
            coeffs.len()
        );
    }
    if let CameraIntrinsic::Manual {
        focal_len_x,
        focal_len_y,
        skew,
        offset_x,
        offset_y,
    } = &args.cam_matrix
    {
        assert_eq!(
            focal_len_x.classify(),
            FpCategory::Normal,
            "Focal length must be a nonzero positive number, found {focal_len_x}"
        );
        assert_eq!(
            focal_len_y.classify(),
            FpCategory::Normal,
            "Focal length must be a nonzero positive number, found {focal_len_y}"
        );
        assert!(!matches!(
            skew.classify(),
            FpCategory::Infinite | FpCategory::Nan
        ));
        assert!(!matches!(
            offset_x.classify(),
            FpCategory::Infinite | FpCategory::Nan
        ));
        assert!(!matches!(
            offset_y.classify(),
            FpCategory::Infinite | FpCategory::Nan
        ));
    }
    assert!(
        args.pnp_ransac_iter_count != 0,
        "RANSAC iteration count must be a nonzero positive number"
    );
}

pub fn camera_relative_to_earth(solution: &PNPRANSACSolution) -> Result<Cartesian3d, MatError> {
    // Load the rotationmatrix and the transformation vector into a combined matrix.
    let rvec = &solution.rvec;
    let tvec = &solution.tvec;

    let mut rot_vec: Cmat<f64> = Cmat::zeros(3, 1).expect("Failed to initialize matrix");

    calib3d::rodrigues_def(rvec, &mut rot_vec).unwrap();

    let rt_vec = solution_to_rt(rvec, tvec)?;

    let rt_mat = OMatrix::<f64, U4, U4>::from_vec(rt_vec);

    

    let camera = OMatrix::<f64, U4, U1>::new(0.0, 0.0, 0.0, 1.0);


    let world = rt_mat.try_inverse().expect("Could not inverse matrix") * camera;

    Ok(Cartesian3d {
        x: world[0] / world[3],
        y: world[1] / world[3],
        z: world[2] / world[3],
    })
}

fn solution_to_rt(rvec: &Cmat<f64>, tvec: &Cmat<f64>) -> Result<Vec<f64>, MatError> {
    let mut rt_vec: Vec<f64> = Vec::new();
    for i in 0..3 * 3 {
        rt_vec.push(rvec.at_2d(i % 3, i / 3)?.to_owned());
        if i % 3 == 2 {
            rt_vec.push(0.0);
        }
    }
    for i in 0..4 {
        if i == 3 {
            rt_vec.push(1.0);
        } else {
            rt_vec.push(tvec.at_2d(i, 0)?.to_owned());
        }
    }
    Ok(rt_vec)
}

pub fn rotation_matrix_to_euler(rotation_matrix: &Cmat<f64>) -> Result<EulerAngles, MatError> {
    let a20 = rotation_matrix.at_2d(2, 0)?;
    let a21 = rotation_matrix.at_2d(2, 1)?;

    let phi = a20.atan2(a21.to_owned());

    let a22 = rotation_matrix.at_2d(2, 2)?;

    let theta = a22.acos();

    let a02 = rotation_matrix.at_2d(0, 2)?;
    let a12 = rotation_matrix.at_2d(1, 2)?;

    let psi = -a02.atan2(a12.to_owned());

    Ok(EulerAngles { roll: theta * -1.0 , pitch: psi, yaw: phi - std::f64::consts::PI })
}

fn mat_to_nalgebra(opencv_mat: &Cmat<f64>) -> Result<OMatrix<f64, Dyn, Dyn>, MatError> {
    let size = opencv_mat
        .mat
        .size()
        .expect("Could not obtain size of matrix.");
    let mut vec = Vec::with_capacity((size.width * size.height) as usize);

    for i in 0..(size.width * size.height) {
        vec.push(opencv_mat.at_2d(i / size.width, i % size.width)?.to_owned());
    }

    let matrix =
        OMatrix::<f64, Dyn, Dyn>::from_row_slice(size.height as usize, size.width as usize, &vec);

    Ok(matrix)
}

pub fn camera_angles(
    solution: &PNPRANSACSolution,
    camera_matrix: &Cmat<f64>,
) -> Result<EulerAngles, MatError> {
    let rt_vec = solution_to_rt(&solution.rvec, &solution.tvec)?;
    type Matrix4x4f64 = OMatrix<f64, U4, U4>;
    let rt_matrix = Matrix4x4f64::from_vec(rt_vec);

    let rotation = rotation_matrix_to_euler(&solution.rvec)?;

    dbg!(&rotation.to_deg());

    let point_3d_normal = Cartesian3d {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    let earth_center = OMatrix::<f64,U4, U1>::new(0.0, 0.0, 0.0, 1.0);
    let earth_center = rt_matrix * earth_center;
    let earth_center = Cartesian3d {
        x: earth_center[0] / earth_center[3],
        y: earth_center[1] / earth_center[3],
        z: earth_center[2] / earth_center[3],
    };

    let cam_coords = Cartesian3d {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };



    let roll = cosinus_relation_solver(
        p2p_distance(
            (earth_center.x, 0.0, earth_center.z),
            (point_3d_normal.x, 0.0, point_3d_normal.z),
        ),
        p2p_distance(
            (earth_center.x, 0.0, earth_center.z),
            (cam_coords.x, 0.0, cam_coords.z),
        ),
        p2p_distance(
            (point_3d_normal.x, 0.0, point_3d_normal.z),
            (cam_coords.x, 0.0, cam_coords.z),
        ),
    );

    let pitch = cosinus_relation_solver(
        p2p_distance(
            (0.0, earth_center.y, earth_center.z),
            (0.0, point_3d_normal.y, point_3d_normal.z),
        ),
        p2p_distance(
            (0.0, earth_center.y, earth_center.z),
            (0.0, cam_coords.y, cam_coords.z),
        ),
        p2p_distance(
            (0.0, point_3d_normal.y, point_3d_normal.z),
            (0.0, point_3d_normal.y, cam_coords.z),
        ),
    );

    let pms = plus_or_minus(&rt_matrix, roll, pitch);

    
    Ok(EulerAngles {roll: roll * pms.0, pitch: pitch * pms.1, yaw: rotation.yaw})
}

fn plus_or_minus(rt_matrix: &OMatrix<f64, U4, U4>, roll: f64, pitch: f64) -> (f64, f64) {
    let point_3d_normal = Cartesian3d {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    let earth_center = OMatrix::<f64,U4, U1>::new(f64::EPSILON * 1000000.0, f64::EPSILON * 1000000.0, 0.0, 1.0);
    let earth_center = rt_matrix * earth_center;
    let earth_center = Cartesian3d {
        x: earth_center[0] / earth_center[3],
        y: earth_center[1] / earth_center[3],
        z: earth_center[2] / earth_center[3],
    };

    let cam_coords = Cartesian3d {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };



    let new_roll = cosinus_relation_solver(
        p2p_distance(
            (earth_center.x, 0.0, earth_center.z),
            (point_3d_normal.x, 0.0, point_3d_normal.z),
        ),
        p2p_distance(
            (earth_center.x, 0.0, earth_center.z),
            (cam_coords.x, 0.0, cam_coords.z),
        ),
        p2p_distance(
            (point_3d_normal.x, 0.0, point_3d_normal.z),
            (cam_coords.x, 0.0, cam_coords.z),
        ),
    );

    let new_pitch = cosinus_relation_solver(
        p2p_distance(
            (0.0, earth_center.y, earth_center.z),
            (0.0, point_3d_normal.y, point_3d_normal.z),
        ),
        p2p_distance(
            (0.0, earth_center.y, earth_center.z),
            (0.0, cam_coords.y, cam_coords.z),
        ),
        p2p_distance(
            (0.0, point_3d_normal.y, point_3d_normal.z),
            (0.0, point_3d_normal.y, cam_coords.z),
        ),
    );

    let mut roll_pm: f64;
    let mut pitch_pm: f64;

    if roll - new_roll > 0.0 {
        roll_pm = -1.0;
    } else {
        roll_pm = 1.0;
    }

    if pitch - new_pitch > 0.0 {
        pitch_pm = -1.0;
    } else {
        pitch_pm = 1.0;
    }

    (roll_pm, pitch_pm)

}

/// ec is earth center\
/// pp is picture point\
/// oc is origo camera
fn cosinus_relation_solver(ec_to_pp: f64, ec_to_oc: f64, pp_to_oc: f64) -> f64 {
    ((ec_to_oc.powi(2) + pp_to_oc.powi(2) - ec_to_pp.powi(2)) / (2.0 * ec_to_oc * pp_to_oc)).acos()
}

fn p2p_distance(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let vector = (a.0 - b.0, a.1 - b.1, a.2 - b.2);

    (vector.0.powi(2) + vector.1.powi(2) + vector.2.powi(2)).sqrt()
}
