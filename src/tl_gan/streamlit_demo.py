import glob
import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import PIL

sys.path.append('src')
sys.path.append('src/model/pggan')
import tl_gan.feature_axis as feature_axis
import tfutil

@st.cache(allow_output_mutation=True)
def load_pg_gan_model():
    """
    Create the tensorflow session.
    """
    print('*** Create TF Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=config)

    print('*** Load GAN Model')
    path_model = './asset_model/karras2018iclr-celebahq-1024x1024.pkl'

    with session.as_default():
        with open(path_model, 'rb') as f:
            G, D, Gs = pickle.load(f)
    return session, G

@st.cache
def load_tl_gan_model():
    """
    Load the linear model (matrix) which maps the feature space
    to the GAN's latent space.
    """
    print('*** Load TL-GAN Model')
    path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'
    pathfile_feature_direction = \
        glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

    with open(pathfile_feature_direction, 'rb') as f:
        feature_direction_name = pickle.load(f)

    feature_direction = feature_direction_name['direction']
    feature_names = feature_direction_name['name']
    num_feature = feature_direction.shape[1]
    feature_lock_status = np.zeros(num_feature).astype('bool')
    feature_direction_disentangled = \
        feature_axis.disentangle_feature_axis_by_idx(
            feature_direction,
            idx_base=np.flatnonzero(feature_lock_status))
    return feature_direction_disentangled, feature_names

def get_default_features(feature_names):
    """
    Return a default dictionary from feature names to feature 
    values. The features are defined on the range [0,100].
    """
    features = dict((name, 49) for name in feature_names)
    return features

@st.cache(hash_funcs={tf.Session : id, tfutil.Network : id}, show_spinner=False)
def generate_image(session, pg_gan_model, tl_gan_model, features, feature_names):
    """
    Converts a feature vector into an image.
    """
    latents = convert_features_to_latent_variables(tl_gan_model, features, feature_names)
    latents = latents.reshape(1, -1)
    dummies = np.zeros([1] + pg_gan_model.input_shapes[1][1:])
    with session.as_default():
        images = pg_gan_model.run(latents, dummies)
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    return images[0]

@st.cache(hash_funcs={tfutil.Network : id})
def convert_features_to_latent_variables(tl_gan_model, features, feature_names):
    """Uses Shaobo's model to convert the feature vector
    (a dictionary from feature names to values) into
    a numpy array consisting of the latent variables."""
    # convert the features from a dict to an array
    feature_values = np.array([features[name] for name in feature_names])

    # renormalize from [0,100] -> [-0.2, 0.2]
    feature_values = (feature_values - 50) / 1000

    # muliply by shaobo's matrix to get the latent variables
    latents = np.dot(tl_gan_model, feature_values)
    return latents


st.title("Demo of Shaobo's GAN")

session, pg_gan_model = load_pg_gan_model()
tl_gan_model, feature_names = load_tl_gan_model()

features = get_default_features(feature_names)
features['Male'] = st.slider('Male', 0, 100, 49, 5)
features['Smiling'] = st.slider('Smiling', 0, 100, 49, 5)
features['Bald'] = st.slider('Bald', 0, 100,49, 5)

image_out = generate_image(session, pg_gan_model, tl_gan_model, features, feature_names)

st.image(image_out, width=400)
