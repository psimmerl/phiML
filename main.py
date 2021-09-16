from _typeshed import FileDescriptor
from modules.models import phiDNN
from modules.data import ROOTParser

flist = ["root1.root", "root2.root"]

dnn_model = phiDNN()
data = ROOTParser(flist)