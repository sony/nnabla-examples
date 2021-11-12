# ONNX Runtime Web demonstration using file format converter

This contains the demonstration to convert nnp (nnabla model) to onnx,
and inference on ONNX Runtime Web.


## start/stop web service

There are two ways to start http server.
Either way, you can access port 8000 as `http://127.0.0.1:8000`.

### web service by python

The following command will start http server on port 8000,
and stop it by `Ctrl-C`.

```bash
cd html && python -m http.server 8000
```

### web service by docker

The following command will start http server on port 8000.
This command prints container ID which is 64bytes string.

```bash
docker run -p 8000:80 -v $(pwd)/html:/usr/share/nginx/html --rm -d nginx:latest
```

To stop web server, call `docker stop` with container ID that is printed by run command.
You can check short container ID by `docker ps -a`.

```bash
docker stop [container ID]
```


## convert resnet-50

This script will download resnet-50 pre-trained model, and convert to onnx.
The converted model will be saved as html/resnet_4_178.onnx.

```bash
./convert-resnet50.sh
```


# Reference

* ONNX Runtime Web GitHub: https://github.com/microsoft/onnxruntime
