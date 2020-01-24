import scenenet_pb2 as sn
import os

data_root_path = 'data/val'
protobuf_path = 'data/scenenet_rgbd_val.pb'

# These functions produce a file path (on Linux systems) to the image given
# a view and render path from a trajectory.  As long the data_root_path to the
# root of the dataset is given.  I.e. to either val or train
def photo_path_from_view(render_path,view):
    photo_path = os.path.join(render_path,'photo')
    image_path = os.path.join(photo_path,'{0}.jpg'.format(view.frame_num))
    return os.path.join(data_root_path,image_path)

def instance_path_from_view(render_path,view):
    photo_path = os.path.join(render_path,'instance')
    image_path = os.path.join(photo_path,'{0}.png'.format(view.frame_num))
    return os.path.join(data_root_path,image_path)

def depth_path_from_view(render_path,view):
    photo_path = os.path.join(render_path,'depth')
    image_path = os.path.join(photo_path,'{0}.png'.format(view.frame_num))
    return os.path.join(data_root_path,image_path)


if __name__ == '__main__':
    trajectories = sn.Trajectories()
    try:
        with open(protobuf_path,'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(data_root_path))
        print('Please ensure you have copied the pb file to the data directory')

    print('Number of trajectories:{0}'.format(len(trajectories.trajectories)))
    for traj in trajectories.trajectories:
        layout_type = sn.SceneLayout.LayoutType.Name(traj.layout.layout_type)
        layout_path = traj.layout.model
        print('='*20)
        print('Render path:{0}'.format(traj.render_path))
        print('Layout type:{0} path:{1}'.format(layout_type,layout_path))
        print('='*20)
        print('')
        print('Number of instances: {0}'.format(len(traj.instances)))
        '''
        The instances attribute of trajectories contains all of the information
        about the different instances.  The instance.instance_id attribute provides
        correspondences with the rendered instance.png files.  I.e. for a given
        trajectory, if a pixel is of value 1, the information about that instance,
        such as its type, semantic class, and wordnet id, is stored here.
        For more information about the exact information available refer to the
        scenenet.proto file.
        '''
        for instance in traj.instances:
            instance_type = sn.Instance.InstanceType.Name(instance.instance_type)
            print('='*20)
            print('Instance id:{0}'.format(instance.instance_id))
            print('Instance type:{0}'.format(instance_type))
            if instance.instance_type != sn.Instance.BACKGROUND:
                print('Wordnet id:{0}'.format(instance.semantic_wordnet_id))
                print('Plain english name:{0}'.format(instance.semantic_english))
            if instance.instance_type == sn.Instance.LIGHT_OBJECT:
                light_type = sn.LightInfo.LightType.Name(instance.light_info.light_type)
                print('Light type:{0}'.format(light_type))
            if instance.instance_type == sn.Instance.RANDOM_OBJECT:
                print('Object info:{0}'.format(instance.object_info))
            print('-'*20)
            print('')
        print('Render path:{0}'.format(traj.render_path))
        '''
        The views attribute of trajectories contains all of the information
        about the rendered frames of a scene.  This includes camera poses,
        frame numbers and timestamps.
        '''
        for view in traj.views:
            print(photo_path_from_view(traj.render_path,view))
            print(depth_path_from_view(traj.render_path,view))
            print(instance_path_from_view(traj.render_path,view))
            print(view)
        break