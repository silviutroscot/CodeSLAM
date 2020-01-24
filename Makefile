all:
	protoc --python_out=./ scenenet.proto

clean:
	$(RM) scenenet_pb2.py
	$(RM) -r __pycache__