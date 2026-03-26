all:
	protoc --python_out=./ scenenet.proto

compile:
	python3 -m py_compile codeslam/*.py scripts/*.py tests/*.py

test:
	python3 -m unittest discover -s tests

clean:
	$(RM) scenenet_pb2.py
	$(RM) -r __pycache__
