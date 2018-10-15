import pickle


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    print(f'Loaded: {filename}')
    return data


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f'Saved: {filename}')