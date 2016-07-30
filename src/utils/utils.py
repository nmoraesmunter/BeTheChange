import cPickle as pickle

def save_pickle(filename, obj):
    with open(filename, 'w') as f:
        f.write(pickle.dumps(obj))

def load_pickle(filename):
    with open(filename, 'r') as f:
        out = pickle.loads(f.read())
    return out

def save_model(model, model_name):
    save_pickle('../app/models/%s.pkl'% model_name, model)

def load_model(model_name):
    model = load_pickle('../app/models/%s.pkl'% model_name)
    return model
