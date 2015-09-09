from chainer import FunctionSet,Variable
import chainer.functions as F

class LeNet(FunctionSet):

    def __init__(self):
        super(LeNet,self).__init__(
            conv1 = F.Convolution2D(1,20,5,stride = 1),
            #bn1   = F.BatchNormalization( 4),
            conv2 = F.Convolution2D(20,50,5,stride = 1),
            fc3 = F.Linear(800,500),
            fc4 = F.Linear(500,10),
)
    def forward(self,x_data,y_data,train=True):
        x = Variable(x_data,volatile=not train)
        t = Variable(y_data,volatile=not train)
        h = F.max_pooling_2d(self.conv1(x),ksize=2,stride=2)
        h = F.max_pooling_2d(self.conv2(h),ksize=2,stride=2)
        h = F.dropout(self.fc3(h))
        h = self.fc4(h)
        return F.softmax_cross_entropy(h,t),F.accuracy(h,t)

