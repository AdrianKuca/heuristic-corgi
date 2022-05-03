import importlib


def load_model_module(model_file):
    spec = importlib.util.spec_from_file_location("model", str(model_file))
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module


def get_model_info(model_module, size):
    model_info = []
    model = model_module.get_model(size)
    model.summary(print_fn=lambda x: model_info.append(x))
    return model_info


def get_model_name(model_module, size):
    model_info = get_model_info(model_module, size)

    return model_info[0].split('"')[1].split('"')[0]
