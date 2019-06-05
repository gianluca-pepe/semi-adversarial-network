from os import path, makedirs, mkdir


def save_weights(model, name):
    paths = RepoPaths()

    filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = str(filename) + ".h5"
    filename = name + '-' + filename
    file_path = path.join(paths.weights, filename)
    model.save_weights(file_path)


class RepoPaths:
    def __init__(self):
        self.root = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))

        self.weights = path.join(self.root, 'weights')

        self.ds = path.join(self.root, 'dataset')
        self.ds_lfw = path.join(self.ds, 'LFW')
        self.ds_celeb = path.join(self.ds, 'celebA')
        self.ds_celeb_generated = path.join(self.ds_celeb, 'generated')

        self.proto = path.join(self.root, 'dataset', 'prototype')

        # CelebA paths
        train = path.join(self.ds_celeb, 'train')
        train_f = path.join(train, 'female')
        train_m = path.join(train, 'male')

        test = path.join(self.ds_celeb, 'test')
        test_f = path.join(test, 'female')
        test_m = path.join(test, 'male')

        valid = path.join(self.ds_celeb, 'valid')
        valid_f = path.join(valid, 'female')
        valid_m = path.join(valid, 'male')

        self.celeba = {
            'train': train,
            'train_female': train_f,
            'train_male': train_m,
            'test': test,
            'test_female': test_f,
            'test_male': test_m,
            'valid': valid,
            'valid_female': valid_f,
            'valid_male': valid_m,
            'generated': {
                'male': self.get_celeba_gen('male'),
                'female': self.get_celeba_gen('female')
            }
        }

        self.lfw = {
            'male': self.get_lfw_paths('male'),
            'female': self.get_lfw_paths('female')
        }

        self.create_folders()

    def get_lfw_paths(self, gender):
        # LFW paths
        or1 = path.join(self.ds_lfw, gender, 'OR_1')
        or2 = path.join(self.ds_lfw, gender, 'OR_2')
        nt = path.join(self.ds_lfw, gender, 'NT')
        op = path.join(self.ds_lfw, gender, 'OP')
        sm = path.join(self.ds_lfw, gender, 'SM')

        lfw = {
            'or1': or1,
            'or2': or2,
            'nt': nt,
            'op': op,
            'sm': sm,
        }

        return lfw

    def get_celeba_gen(self, gender):
        # celeba paths
        nt = path.join(self.ds_celeb_generated, gender, 'NT')
        op = path.join(self.ds_celeb_generated, gender, 'OP')
        sm = path.join(self.ds_celeb_generated, gender, 'SM')

        celeba_gen = {
            'nt': nt,
            'op': op,
            'sm': sm,
        }

        return celeba_gen

    def create_folders(self):
        try:
            mkdir(self.weights)
        except OSError:
            print()

        try:
            mkdir(self.ds)
        except OSError:
            print()

        try:
            mkdir(self.ds_celeb)
        except OSError:
            print()

        try:
            mkdir(self.ds_lfw)
        except OSError:
            print()

        try:
            for key in self.celeba.keys():
                if key == 'generated':
                    for gender in self.celeba[key].keys():
                        for type in self.celeba[key][gender].keys():
                            makedirs(self.celeba[key][gender][type])
                else:
                    makedirs(self.celeba[key])
        except OSError:
            print()

        try:
            for gender in self.lfw.keys():
                for key in self.lfw[gender].keys():
                    makedirs(self.lfw[gender][key])
        except OSError:
            print()
