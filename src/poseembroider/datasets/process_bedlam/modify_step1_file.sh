#!/bin/bash

echo "Subject to README at https://github.com/pixelite1201/BEDLAM/tree/master/data_processing/ReadMe.md i.e., modification of code limited to supporting 'research for tracking, multi-person pose estimation etc.'"

# This script modifies 'df_full_body.py' to match 'process_bedlam_step1.py'

sed -i '13,13d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '16,20d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '76,76d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '82,82d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '88,88d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '94,95d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '96,128d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '130,140d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '142,142d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '210,210d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '211,211d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '213,213d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '223,230d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '253,253d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '256,257d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '258,259d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '259,260d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '265,265d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '297,297d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '302,303d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '304,306d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '307,308d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '316,316d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '320,322d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '323,324d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '329,329d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '333,337d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '346,347d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '347,347d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '351,351d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '358,358d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '378,386d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '386,387d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '394,395d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '395,395d' datasets/process_bedlam/process_bedlam_step1.py
sed -i '1i """
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '2i Modified version of https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/df_full_body.py
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '3i """
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '4i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '6i import re
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '17i import streamlit as st
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '19i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '20i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '21i import poseembroider.config as config
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '22i from poseembroider.datasets.base_dataset import padd_and_resize_keeping_aspect_ratio
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '23i from poseembroider.augmentations import transform_for_visu
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '27i SMPLX_DIR = config.SMPLX_BODY_MODEL_PATH
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '28i IMAGE_FOLDER = config.BEDLAM_IMG_DIR # NOTE, the '\''split'\'' is added later 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '29i SMPLX_GT_FOLDER = os.path.join(config.BEDLAM_RAW_DATA_DIR, "neutral_ground_truth_motioninfo")
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '30i CSV_FOLDER = os.path.join(config.BEDLAM_RAW_DATA_DIR, '\''csv'\'')
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '31i OUTPUT_FOLDER = os.path.join(config.BEDLAM_PROCESSED_DIR, "processed_npz")
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '32i if not os.path.isdir(OUTPUT_FOLDER):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '33i \    os.makedirs(OUTPUT_FOLDER)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '34i UTIL_FOLDER = config.BEDLAM_RAW_DATA_DIR
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '35i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '36i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '37i VISUALIZE = False
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '38i st.set_page_config(layout="wide")
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '39i cols = st.columns(8)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '40i COL_IND = 0
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '41i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '42i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '43i # VISU
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '44i ################################################################################
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '45i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '46i import roma
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '47i import trimesh
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '48i import pyrender
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '49i from body_visualizer.mesh.mesh_viewer import MeshViewer
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '50i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '51i # meshviewer
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '52i imw, imh = 1600, 1600
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '53i mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '54i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '55i # body model
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '56i @st.cache_resource
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '57i def get_body_model():
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '58i \    body_model_smplx = smplx.create(SMPLX_DIR,
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '59i \                '\''smplx'\'',
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '60i \                gender='\''neutral'\'',
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '61i \                num_betas=11,
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '62i \                use_pca = False,
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '63i \                flat_hand_mean = True,
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '64i \                batch_size=1)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '65i \    body_model_smplx.eval()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '66i \    body_model_smplx.to('\''cpu'\'')
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '67i \    return body_model_smplx
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '68i body_model_smplx = get_body_model()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '69i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '70i from PIL import Image
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '71i basic_image_transform =  padd_and_resize_keeping_aspect_ratio()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '72i tfv = transform_for_visu(unormalize=False)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '73i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '74i def convert(image):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '75i \    return torch.permute(image, (1,2,0)).numpy()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '76i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '77i def load_image(image_path, bbox):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '78i \    # open image
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '79i \    image = Image.open(image_path)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '80i \    # convert image
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '81i \    if image.mode != '\''RGB'\'':
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '82i \        image = image.convert('\''RGB'\'')
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '83i \    # crop image to the bounding box of the human
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '84i \    image = image.crop(bbox)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '85i \    # return padded & resized tensor
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '86i \    image = basic_image_transform(image)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '87i \    return image
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '88i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '89i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '90i def pose_to_image(pose_data, body_model_smplx, color=[0.4, 0.8, 1.0]):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '91i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '92i \    pose_data = pose_data.reshape(1,-1)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '93i \    bm = body_model_smplx(
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '94i \        global_orient=pose_data[:,:3],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '95i \        body_pose=pose_data[:,3:66],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '96i \        jaw_pose=pose_data[:,66:69],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '97i \        leye_pose=pose_data[:,69:72],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '98i \        reye_pose=pose_data[:,72:75],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '99i \        left_hand_pose=pose_data[:,75:120],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '100i \        right_hand_pose=pose_data[:,120:165]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '101i \    )
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '102i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '103i \    vertices = bm.vertices[0].detach().numpy()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '104i \    body_mesh = trimesh.Trimesh(vertices=vertices, faces=body_model_smplx.faces, vertex_colors=np.tile(color+[1.], (vertices.shape[0], 1)))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '105i \    mv.set_static_meshes([body_mesh], [np.eye(4)])
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '106i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '107i \    # add axis
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '108i \    transl_orig = torch.tensor([vertices[:,0].max()*1.2, vertices[:,1].min(), 0])
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '109i \    t = get_homogeneous(rotvec=None, transl=transl_orig)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '110i \    axis_w = trimesh.creation.axis(origin_size=0.04, transform=t.numpy())
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '111i \    axis_w = pyrender.Mesh.from_trimesh(axis_w, smooth=False)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '112i \    mv.scene.add(axis_w, '\''static-mesh-1000'\'') # must have the '\''static-mesh'\'' name so that it is properly removed by MeshViewer for future renderings
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '113i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '114i \    tt = bm.joints[0][0].detach()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '115i \    tt[0] = vertices[:,0].max()*1.2
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '116i \    t = get_homogeneous(rotvec=pose_data[:,:3], transl=tt)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '117i \    axis_b = trimesh.creation.axis(origin_size=0.04, transform=t.numpy(), origin_color=np.array([0.4, 0.4, 1.0]))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '118i \    axis_b = pyrender.Mesh.from_trimesh(axis_b, smooth=False)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '119i \    mv.scene.add(axis_b, '\''static-mesh-1001'\'') # must have the '\''static-mesh'\'' name so that it is properly removed by MeshViewer for future renderings
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '120i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '121i \    # projections of x_smpl and z_smpl
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '122i \    x_smpl, y_smpl, z_smpl = get_axis_coords(pose_data[0,:3].view(1,3).clone())
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '123i \    x_axis, y_axis, z_axis = get_axis_coords()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '124i \    x_smpl_proj = x_smpl - (x_smpl @ y_axis)*y_axis
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '125i \    z_smpl_proj = z_smpl - (z_smpl @ y_axis)*y_axis
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '126i \    v1 = trimesh.creation.cylinder(segment=np.concatenate([transl_orig.numpy(), (transl_orig+x_smpl_proj).numpy()]).reshape(2,3), radius=0.04/5.0)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '127i \    v2 = trimesh.creation.cylinder(segment=np.concatenate([transl_orig.numpy(), (transl_orig+z_smpl_proj).numpy()]).reshape(2,3), radius=0.04/5.0)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '128i \    for i,v in enumerate([v1,v2]):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '129i \        v = pyrender.Mesh.from_trimesh(v, smooth=False)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '130i \        mv.scene.add(v, '\''static-mesh-100'\''+str(2+i))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '131i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '132i \    img = mv.render(render_wireframe=False)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '133i \    img = np.array(Image.fromarray(img)) # img of shape (H, W, 3)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '134i \    return img
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '135i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '136i # MAIN
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '137i ################################################################################
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '138i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '139i # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '140i def get_homogeneous(rotvec=None, transl=None):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '141i \    r = torch.eye(4)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '142i \    if rotvec is not None:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '143i \        r[:3,:3] = roma.rotvec_to_rotmat(rotvec.view(1,3))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '144i \    if transl is not None:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '145i \        r[:3,3] = transl
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '146i \    return r
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '147i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '148i # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '149i def get_homogeneous_np(rotvec=None, transl=None):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '150i \    return get_homogeneous(rotvec=rotvec, transl=transl).numpy()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '151i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '152i # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '153i def get_axis_coords(rotvec=None, axis_length=1):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '154i \    """
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '155i \    get the axis coords from the provided rotvec
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '156i \    inspired from trimesh/creation.py > axis
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '157i \    can be found here:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '158i \    .../conda/envs/<name_environment>/lib/python3.8/site-packages/trimesh/creation.py
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '159i \    """
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '160i \    # (1) build axis
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '161i \    t = get_homogeneous_np(rotvec=rotvec)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '162i \    translation = get_homogeneous_np(transl=torch.tensor([0, 0, axis_length]))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '163i \    # ---
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '164i \    z_axis = t.dot(translation)[:3,3]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '165i \    # ---
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '166i \    rot = get_homogeneous_np(rotvec=torch.tensor([1,0,0])*(-torch.pi/2))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '167i \    y_axis = t.dot(rot).dot(translation)[:3,3]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '168i \    # ---
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '169i \    rot = get_homogeneous_np(rotvec=torch.tensor([0,1,0])*(torch.pi/2))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '170i \    x_axis = t.dot(rot).dot(translation)[:3,3]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '171i \    # (2) convert to torch and normalize
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '172i \    all_axis = [x_axis, y_axis, z_axis]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '173i \    for i, axis in enumerate(all_axis):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '174i \        axis = torch.from_numpy(axis)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '175i \        axis /= torch.norm(axis)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '176i \        all_axis[i] = axis
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '177i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '178i \    return all_axis
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '179i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '180i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '181i # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '182i def normalize_pose(pose):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '183i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '184i \    with torch.no_grad():
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '185i \        pose_data = convert_to_torch(pose).reshape(1,-1)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '186i \        bm = body_model_smplx(
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '187i \            global_orient=pose_data[:,:3],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '188i \            body_pose=pose_data[:,3:66],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '189i \            # jaw_pose=pose_data[:,66:69],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '190i \            # leye_pose=pose_data[:,69:72],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '191i \            # reye_pose=pose_data[:,72:75],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '192i \            left_hand_pose=pose_data[:,75:120],
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '193i \            right_hand_pose=pose_data[:,120:165]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '194i \        )
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '195i \        j = bm.joints.detach()[0] # shape (J, 3)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '196i \        body_hips = (j[1] - j[2]).view(1,3) # shape (1,3)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '197i \        body_hips = body_hips / torch.norm(body_hips, dim=1, keepdim=True)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '198i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '199i \    # define the current 2d position of the points to align
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '200i \    # and the target 2d position (~ direction) those points should have
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '201i \    p2d_L, t2d_L = j[1, [0,2]].view(1,2), torch.tensor([1.0,0]).view(1,2) # components X,Z
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '202i \    p2d_R, t2d_R = j[2, [0,2]].view(1,2), torch.tensor([-1.0,0]).view(1,2) # components X,Z
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '203i \    points_2d = torch.cat((p2d_L, p2d_R)).view(1,len(p2d_L)+len(p2d_R),2)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '204i \    targets_2d = torch.cat((t2d_L, t2d_R)).view(1,len(p2d_L)+len(p2d_R),2)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '205i \    # compute rotation
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '206i \    R2d, t2d = roma.rigid_points_registration(points_2d, targets_2d)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '207i \    # convert 2d rot mat to 3d rot mat
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '208i \    R3d = torch.eye(3,3).view(1,3,3)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '209i \    R3d[0,0,0] = R2d[0,0,0]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '210i \    R3d[0,2,0] = R2d[0,1,0]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '211i \    R3d[0,0,2] = R2d[0,0,1]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '212i \    R3d[0,2,2] = R2d[0,1,1]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '213i \    # apply rotation
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '214i \    initial_rotation = roma.rotvec_composition([roma.rotmat_to_rotvec(R3d), convert_to_torch(pose[:3]).view(1,3)])
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '215i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '216i \    pose[:3] = initial_rotation.view(3).numpy()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '217i \    return pose
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '218i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '219i # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '220i def theta2str(theta):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '221i \    return str(round((theta*180/torch.pi).item(), 2))+"°"
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '222i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '223i # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '224i def vec2str(vec):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '225i \    vec = vec.view(3)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '226i \    return "[" + ", ".join([str(round(vv, 3)) for vv in vec.tolist()]) + "]"
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '227i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '228i # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '229i def get_inside_points(points, max_x=1280, max_y=720):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '230i \    # only consider the points that are inside the boundaries
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '231i \    # (ie. whose "both" coordinates are simulatenously inside the boundaries)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '232i \    inside_point_ids = (points[:,0]<max_x) * (points[:,1]<max_y)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '233i \    points = points[inside_point_ids,:]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '234i \    return points
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '235i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '236i # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '237i def get_tight_bbox(points, factor=1., max_x=1280, max_y=720): # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '238i \    assert len(points.shape) == 2, f"Wrong shape, expected two-dimensional array. Got shape {points.shape}"
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '239i \    assert points.shape[1] == 2
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '240i \    try:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '241i \        # print("proj_verts_ min(0), min(1):", points[:,0].min(), points[:,1].min())
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '242i \        # print("proj_verts_ max(0), max(1):", points[:,0].max(), points[:,1].max())
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '243i \        inside_points = get_inside_points(points, max_x, max_y)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '244i \        # print("proj_verts_ min(0), min(1):", inside_points[:,0].min(), inside_points[:,1].min())
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '245i \        # print("proj_verts_ max(0), max(1):", inside_points[:,0].max(), inside_points[:,1].max())
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '246i \        x1, x2 = inside_points[:,0].min(), inside_points[:,0].max()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '247i \        y1, y2 = inside_points[:,1].min(), inside_points[:,1].max()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '248i \    except ValueError:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '249i \        # import traceback
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '250i \        # traceback.print_exc()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '251i \        # import pdb; pdb.set_trace()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '252i \        return None
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '253i \    cx, cy = (x2 + x1) / 2., (y2 + y1) / 2.
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '254i \    sx, sy = np.abs(x2 - x1), np.abs(y2 - y1)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '255i \    sx, sy = int(factor * sx), int(factor * sy)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '256i \    x1, y1 = int(cx - sx / 2.), int(cy - sy / 2.)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '257i \    x2, y2 = int(cx + sx / 2.), int(cy + sy / 2.)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '258i \    return [x1,y1,x2,y2]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '259i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '260i # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '261i def get_tightplus_bbox(corners, scale_factor=1.1, max_x=1280, max_y=720): # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '262i \    x1,y1,x2,y2 = corners # tight bbox
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '263i \    # using as offset for all margins a percentage of the diagonal
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '264i \    offset = (scale_factor-1) * ((x2-x1)**2 + (y2-y1)**2)**0.5 / 2
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '265i \    x1 -= offset
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '266i \    x2 += offset
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '267i \    y1 -= offset
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '268i \    y2 += offset
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '269i \    # sanitize
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '270i \    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '271i \    x1, x2 = max(x1, 0), min(x2, max_x)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '272i \    y1, y2 = max(y1, 0), min(y2, max_y)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '273i \    return [x1, y1, x2, y2]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '274i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '335i smplx_model_male = smplx.create(SMPLX_DIR, model_type='\''smplx'\'',
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '342i smplx_model_female = smplx.create(SMPLX_DIR, model_type='\''smplx'\'',
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '349i smplx_model_neutral = smplx.create(SMPLX_DIR, model_type='\''smplx'\'',
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '356i downsample_mat = pickle.load(open(f'\''{UTIL_FOLDER}/downsample_mat_smplx.pkl'\'', '\''rb'\'')) # get the file from here: https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/downsample_mat_smplx.pkl
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '357i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '358i # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '359i def convert_to_torch(v):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '360i \    if type(v) is np.ndarray:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '361i \        return torch.tensor(v)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '362i \    # return v # assume type(v) is torch.Tensor
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '363i \    return v.clone().detach()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '364i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '365i # CHANGED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '367i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '368i \    with torch.no_grad():
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '369i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '370i \        if gender == "male":
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '371i \            m = smplx_model_male
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '372i \        elif gender == "female":
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '373i \            m = smplx_model_female
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '374i \        elif gender == "neutral":
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '375i \            m = smplx_model_neutral
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '376i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '377i \        betas = convert_to_torch(betas)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '378i \        poses = convert_to_torch(poses)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '379i \        trans = convert_to_torch(trans)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '380i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '381i \        model_out = m(betas=betas.unsqueeze(0).float(),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '382i \                        global_orient=poses[:3].unsqueeze(0).float(),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '383i \                        body_pose=poses[3:66].unsqueeze(0).float(),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '384i \                        left_hand_pose=poses[75:120].unsqueeze(0).float(),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '385i \                        right_hand_pose=poses[120:165].unsqueeze(0).float(),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '386i \                        jaw_pose=poses[66:69].unsqueeze(0).float(),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '387i \                        leye_pose=poses[69:72].unsqueeze(0).float(),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '388i \                        reye_pose=poses[72:75].unsqueeze(0).float(),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '389i \                        transl=trans.unsqueeze(0))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '425i # def visualize(image_path, verts, focal_length, smpl_faces):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '426i # 	img = cv2.imread(image_path)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '427i # 	if rotate_flag:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '428i # 		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '429i # 	h, w, c = img.shape
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '430i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '431i # 	renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '432i # 						faces=smpl_faces)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '433i # 	front_view = renderer.render_front_view(verts.unsqueeze(0).detach().cpu().numpy(),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '434i # 											bg_img_rgb=img[:, :, ::-1].copy())
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '435i \    # cv2.imwrite(image_path.split('\''/'\'')[-4]+image_path.split('\''/'\'')[-1], front_view[:, :, ::-1])
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '448i \    st.pyplot(fig)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '515i \    # from skimage.transform import rotate, resize
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '518i \        new_img = cv2.rotate(new_img, rot) # scipy.misc.imrotate(new_img, rot)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '522i \    new_img = cv2.resize(new_img, res) # scipy.misc.imresize(new_img, res)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '526i # CHANGED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '534i \    # renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '535i \    #\    \    \    \    \     faces=smpl_faces)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '536i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '537i \    # front_view = renderer.render_front_view(verts.unsqueeze(0).detach().cpu().numpy(),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '538i \    #\    \    \    \    \    \    \    \    \    \     bg_img_rgb=img[:, :, ::-1].copy())
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '539i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '540i \    # img, crop_img = crop(front_view[:, :, ::-1], center, scale, res=(224, 224))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '541i \    img, crop_img = crop(img, center, scale, res=(224, 224))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '542i \    # cv2.imwrite(image_path.split('\''/'\'')[-1], crop_img)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '543i \    st.image(crop_img/255.)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '572i \    global COL_IND
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '573i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '574i \    # ADDED:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '575i \    if "30fps" in image_folder:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '576i \        actual_fps = 30
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '577i \    else:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '578i \        # NOTE: folders which are marked with `_{x}fps` only have images every `30/x`
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '579i \        # so no need to further filter out some images + need to adapt `img_ind`
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '580i \        actual_fps = int(re.search(r'\''\\d*fps'\'', image_folder).group(0).replace('\''fps'\'', '\'''\''))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '581i \        assert fps == actual_fps # could also work if fps is a submultiple of actual_fps
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '586i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '588i \        # CHANGED to account for actual_fps
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '589i \        if fps == 6 and actual_fps == 30:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '592i \        elif fps == 6 and actual_fps == 6:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '593i \            # get back to the original img_ind
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '594i \            img_ind = img_ind * 5
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '596i \            ValueError
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '597i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '598i \        smplx_param_ind = img_ind+start_frame
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '599i \        cam_ind = img_ind
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '603i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '607i \        motion_info = smplx_param_orig.get('\''motion_info'\'', '\'''\'') # CHANGED (case of missing key)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '632i \        root_cam_ = c_global_orient # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '636i \        root_world_ = w_global_orient # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '637i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '638i \        # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '639i \        pose = normalize_pose(pose)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '640i \        _, joints3d_can_ = get_smplx_vertices(pose, torch.zeros(len(beta)), torch.zeros(3), '\''neutral'\'') # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '641i \        joints3d_can_ = joints3d_can_.detach().cpu().numpy()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '647i \        proj_verts_ = project(vertices3d_downsample, torch.tensor(cam_trans), CAM_INT) # (437, 3)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '653i \        if num_vis_joints < 3:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '655i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '656i \        # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '657i \        # visu projected vertices
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '658i \        if False and VISUALIZE:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '659i \            points = proj_verts_[:,:2].astype(int)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '660i \            points = get_inside_points(points)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '661i \            im = np.zeros((720,1280))
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '662i \            im[points[:,1], points[:,0]] = 1
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '663i \            st.image(im, caption="sampled vertices")
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '664i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '665i \        # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '666i \        # (only consider points within the image frame)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '667i \        tight_bbox = get_tight_bbox(proj_verts_[:,:2], factor=1.) # ADDED # factor to 1 to get a tight bbox ; # take [:,:2] as it was projected in 2D and the 3rd dim does not mean anything
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '668i \        if tight_bbox is None:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '669i \            continue
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '670i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '671i \        if VISUALIZE:
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '672i \            print("\\n\\n")
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '673i \            print(motion_info)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '674i \            # st.write(motion_info[2])
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '675i \            # cols = st.columns(2)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '676i \            bbox_ = tight_bbox
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '677i \            # bbox_ = get_tightplus_bbox(tight_bbox, scale_factor=1)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '678i \            # cols[0].image(convert(tfv(load_image(image_path, [0, 0, 1280, 720]))), caption="full image")
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '679i \            # cols[0].image(convert(tfv(load_image(image_path, bbox_))), caption="human crop")
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '680i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '681i \            cols[COL_IND].image(pose_to_image(convert_to_torch(pose).view(-1, 3),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '682i \                                            body_model_smplx), caption="(base)")
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '683i \            cols[COL_IND].image(pose_to_image(convert_to_torch(pose).view(-1, 3),
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '684i \                                        body_model_smplx), caption="normalized (base)")
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '685i \            COL_IND = COL_IND + 1
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '686i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '687i \            # visualize_2d(image_path, joints2d)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '688i \            # visualize_2d(image_path, proj_verts_)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '689i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '690i \            # verts_cam2 = vertices3d.detach().cpu().numpy() + cam_trans
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '691i \            # visualize(image_path, torch.tensor(verts_cam2), CAM_INT[0][0], smplx_model_male.faces)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '692i \            # visualize_crop(image_path, center, scale, torch.tensor(verts_cam2) , CAM_INT[0][0], smplx_model_male.faces)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '697i \        base_poses.append(pose) # ADDED (with the original root orientation, normalized)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '698i \        # poses_cam.append(pose_cam)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '699i \        root_cam.append(root_cam_) # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '700i \        # poses_world.append(pose_world)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '701i \        root_world.append(root_world_) # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '702i \        tight_bboxes.append(tight_bbox) # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '703i \        trans_project.append(cam_trans) # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '704i \        joints3d_can.append(joints3d_can_) # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '714i \        subs.append(sub) # identity
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '719i \    parser.add_argument('\''--split'\'', type=str, choices=('\''training'\'', '\''validation'\'', '\''test'\''), default='\''validation'\'')
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '720i \    parser.add_argument('\''--img_folder'\'', type=str, default=IMAGE_FOLDER)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '721i \    parser.add_argument('\''--output_folder'\'', type=str, default=OUTPUT_FOLDER)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '722i \    parser.add_argument('\''--smplx_gt_folder'\'', type=str, default=SMPLX_GT_FOLDER)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '726i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '727i \    print("\\n\\n\\n\\n####################################")
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '728i \    print(f"Processing the {args.split} split.")
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '729i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '730i \    base_image_folder = os.path.join(args.img_folder, args.split) # CHANGED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '731i \    output_folder = os.path.join(args.output_folder, args.split) # CHANGED 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '738i \    image_folders = csv.reader(open(f'\''./custom_bedlam_scene_names.csv'\'', '\''r'\'')) # File to parse folders
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '739i \    # NOTE: ADDED a 3rd field about split information
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '740i \    # NOTE: ADDED a 4th field about the fps of available image folders (3 files have 30fps, all others are 6fps)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '744i \    csv_dict = {}
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '746i \    for row in image_folders: 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '747i \        if row[2] == args.split: # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '748i \            image_dict[row[1]] = os.path.join(base_image_folder, row[0]+f'\''_{row[3]}/png'\'') # CHANGED (added fps info)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '749i \            npz_dict[row[1]] = os.path.join(output_folder, row[0]+f'\''_{fps}fps.npz'\'') # CHANGED (added fps info)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '750i \            csv_dict[row[1]] = os.path.join(CSV_FOLDER, row[0]) # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '751i 
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '752i \    for k, v in image_dict.items():
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '766i \        # image_folder_base = v
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '767i \        # base_folder = v.replace('\''/png'\'','\'''\'')
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '768i \        # base_folder = base_folder.replace(base_image_folder, CSV_FOLDER) # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '769i         
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '771i \        print("outfile:", outfile)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '772i \        if os.path.isfile(outfile):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '773i \            print(f'\''{outfile} EXISTS. WORKING ON THE NEXT FILE.'\'')
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '774i \            continue
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '775i         
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '776i \        csv_path = os.path.join(csv_dict[k], '\''be_seq.csv'\'')
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '781i \        cam_csv_base = os.path.join(csv_dict[k], '\''ground_truth/camera'\'')
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '786i \        base_poses, root_cam, root_world, tight_bboxes, trans_project, joints3d_can =  [], [], [], [], [], [] # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '790i \        for idx, comment in tqdm.tqdm(enumerate(csv_data['\''Comment'\''])):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '811i \                data_path = os.path.join(gt_smplx_folder, person_id, sequence_id, '\''motion_seq.npz'\'')
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '812i \                if os.path.isfile(data_path):
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '813i \                    smplx_param_orig = np.load(data_path)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '814i \                    gender_sub = smplx_param_orig['\''gender_sub'\''].item()
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '815i \                    image_folder = os.path.join(image_dict[k], seq_name)
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '816i \                    X = csv_data['\''X'\''][idx]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '817i \                    Y = csv_data['\''Y'\''][idx]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '818i \                    Z = csv_data['\''Z'\''][idx]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '819i \                    trans_body = [X, Y, Z]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '820i \                    body_yaw_ = csv_data['\''Yaw'\''][idx]
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '821i \                    get_params(image_folder, fl, start_frame, gender_sub, smplx_param_orig, trans_body, body_yaw_, cam_x, cam_y, cam_z, fps, person_id, cam_pitch_=cam_pitch_, cam_roll_=cam_roll_, cam_yaw_=cam_yaw_  )
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '830i \            base_poses=base_poses, # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '831i \            # pose_cam=poses_cam,
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '832i \            root_cam=root_cam, # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '833i \            # pose_world=poses_world,
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '834i \            root_world=root_world, # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '835i \            tight_bboxes=tight_bboxes, # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '836i \            trans_project=trans_project, # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '837i \            joints3d_can=joints3d_can, # ADDED
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '846i \            # motion_info=motion_infos,
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '847i \            sub=subs
' datasets/process_bedlam/process_bedlam_step1.py
sed -i '$ a \        ' datasets/process_bedlam/process_bedlam_step1.py
