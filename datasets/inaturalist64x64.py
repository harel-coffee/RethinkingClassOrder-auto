import os
import random

import numpy as np
import tensorflow as tf

from datasets.dataset import Dataset
from utils import imagenet_preprocessing as imagenet_preprocessing

try:
    import cPickle
except:
    import _pickle as cPickle

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DIR = '/data/hechen/inaturalist2017/prl22_selected/bbox_ver_64x64/train_classified'
TEST_DIR = '/data/hechen/inaturalist2017/prl22_selected/bbox_ver_64x64/val_classified'


def _map_fn(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.ensure_shape(img, (64, 64, 3))
    img = img - tf.broadcast_to(imagenet_preprocessing.CHANNEL_MEANS, tf.shape(img))
    return img, label


class INat64x64(Dataset):
    NUM_CLASSES = 1000
    NUM_MAX_TRAIN_SAMPLES_PER_CLASS = 332
    NUM_MAX_TEST_SAMPLES_PER_CLASS = 40

    def __init__(self, args):
        super().__init__(args)

        # class_names = ['Amphibia_Plethodon cinereus', 'Actinopterygii_Lepomis gibbosus', 'Mammalia_Lepus californicus',
        #                'Mollusca_Ariolimax columbianus', 'Mammalia_Lepus europaeus', 'Arachnida_Uroctonus mordax',
        #                'Insecta_Haematopis grataria', 'Arachnida_Phidippus johnsoni', 'Reptilia_Sceloporus consobrinus',
        #                'Aves_Aegithalos caudatus', 'Reptilia_Ctenosaura similis', 'Aves_Acridotheres tristis',
        #                'Mammalia_Tamias striatus', 'Animalia_Scolopendra polymorpha', 'Insecta_Abaeis nicippe',
        #                'Animalia_Anthopleura sola', 'Actinopterygii_Lepomis cyanellus',
        #                'Mammalia_Odocoileus hemionus californicus', 'Reptilia_Pantherophis alleghaniensis',
        #                'Animalia_Callinectes sapidus', 'Insecta_Dythemis fugax', 'Actinopterygii_Zanclus cornutus',
        #                'Amphibia_Hyla chrysoscelis', 'Mollusca_Cryptochiton stelleri', 'Aves_Junco hyemalis oreganus',
        #                'Arachnida_Leucauge venusta', 'Reptilia_Pantherophis emoryi', 'Animalia_Physalia physalis',
        #                'Aves_Cassiculus melanicterus', 'Insecta_Polistes dominula', 'Mammalia_Sylvilagus floridanus',
        #                'Reptilia_Sceloporus undulatus', 'Insecta_Danaus gilippus', 'Insecta_Argia moesta',
        #                'Amphibia_Hyla versicolor', 'Actinopterygii_Salvelinus fontinalis',
        #                'Mollusca_Hermissenda crassicornis', 'Actinopterygii_Micropterus salmoides',
        #                'Actinopterygii_Oncorhynchus mykiss', 'Mollusca_Hermissenda opalescens',
        #                'Insecta_Apis mellifera', 'Insecta_Olla v-nigrum', 'Mammalia_Lontra canadensis',
        #                'Animalia_Aurelia aurita', 'Actinopterygii_Lepomis macrochirus',
        #                'Animalia_Dermasterias imbricata', 'Mollusca_Diaulula sandiegensis',
        #                'Arachnida_Centruroides vittatus', 'Actinopterygii_Cyprinus carpio', 'Reptilia_Uta stansburiana',
        #                'Reptilia_Trachemys scripta elegans', 'Amphibia_Lithobates sphenocephalus',
        #                'Arachnida_Nephila clavipes', 'Mammalia_Tamiasciurus douglasii',
        #                'Actinopterygii_Pterois volitans', 'Mollusca_Doris montereyensis', 'Arachnida_Peucetia viridans',
        #                'Aves_Sturnus vulgaris', 'Animalia_Pollicipes polymerus', 'Amphibia_Gastrophryne carolinensis',
        #                'Mollusca_Triopha catalinae', 'Aves_Oreothlypis celata', 'Amphibia_Lithobates sylvaticus',
        #                'Mollusca_Phidiana hiltoni', 'Amphibia_Lithobates pipiens', 'Aves_Motacilla cinerea',
        #                'Reptilia_Thamnophis marcianus', 'Arachnida_Gasteracantha cancriformis',
        #                'Reptilia_Micrurus tener', 'Animalia_Cancer productus', 'Aves_Larus californicus',
        #                'Amphibia_Lithobates catesbeianus', 'Arachnida_Latrodectus geometricus',
        #                'Mollusca_Leukoma staminea', 'Amphibia_Rana dalmatina', 'Mammalia_Ursus americanus',
        #                'Animalia_Pisaster ochraceus', 'Aves_Quiscalus quiscula', 'Insecta_Plathemis lydia',
        #                'Mammalia_Procyon lotor', 'Arachnida_Argiope aurantia']
        # self.label_names_dict = {c: c for c in class_names}
        # self.label_names_dict = lambda c: c

    def aug_fn(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.pad_to_bounding_box(image, 4, 4, 64 + 4 * 2, 64 + 4 * 2)
        image = tf.image.random_crop(image, [64, 64, 3])
        return image, label

    def preprocess(self, images, **kwargs):
        images = images.astype(np.float32)
        if images.shape[1] == 3:
            images = images.transpose((0, 2, 3, 1))
        images = images - np.array(imagenet_preprocessing.CHANNEL_MEANS, dtype=np.float32)
        return images

    def load_train(self, wnids, order_wnid, num_samples=-1, parallel_calls=True, random_selection=False,
                   use_shuffle=False):
        filenames = []
        labels = []
        for wnid in wnids:
            sub_dir = os.path.join(TRAIN_DIR, wnid)
            files_cur_cl = [os.path.join(sub_dir, filename) for filename in os.listdir(sub_dir)]
            if num_samples > 0:
                if random_selection:
                    random_indices = list(range(len(files_cur_cl)))
                    random.shuffle(random_indices)
                    files_cur_cl = [files_cur_cl[idx] for idx in random_indices[:num_samples]]
                else:
                    files_cur_cl = files_cur_cl[:num_samples]
            filenames.extend(files_cur_cl)
            labels.extend([order_wnid.index(wnid)] * len(files_cur_cl))

        if use_shuffle:
            indices = list(range(len(filenames)))
            random.shuffle(indices)
            filenames = [filenames[i] for i in indices]
            labels = [labels[i] for i in indices]

        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        if parallel_calls:
            dataset = tf.data.Dataset.from_tensor_slices((filenames, labels)).map(_map_fn,
                                                                                  num_parallel_calls=AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((filenames, labels)).map(_map_fn)

        return dataset

    def load_test(self, wnids, order_wnid, num_samples=-1):
        filenames = []
        labels = []
        for wnid in wnids:
            sub_dir = os.path.join(TEST_DIR, wnid)
            files_cur_cl = [os.path.join(sub_dir, filename) for filename in os.listdir(sub_dir)]
            if num_samples > 0:
                files_cur_cl = files_cur_cl[:num_samples]
            filenames.extend(files_cur_cl)
            labels.extend([order_wnid.index(wnid)] * len(files_cur_cl))

        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels)).map(_map_fn, num_parallel_calls=AUTOTUNE)
        return dataset
