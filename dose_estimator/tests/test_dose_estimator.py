from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc.split(": ")[2].split(",")[0] for x in local_device_protos if x.device_type == 'GPU']

def test_gpu():
    assert get_available_gpus() == ["GeForce GTX 960M"]
