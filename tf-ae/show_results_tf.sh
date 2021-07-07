echo "Resnet-18: "
echo -n "batch size = 1: "
grep Averager logs/resnet18-1-tf.log
echo -n "batch size = 16: "
grep Averager logs/resnet18-16-tf.log 
echo "CSRNet: "
echo -n "batch size = 1: "
grep Averager logs/dilated-1-tf.log 
echo -n "batch size = 16: "
grep Averager logs/dilated-16-tf.log 
echo "Inception-V3: "
echo -n "batch size = 1: "
grep Averager logs/inception_v3-1-tf.log 
echo -n "batch size = 16: "
grep Averager logs/inception_v3-16-tf.log 
echo "Bert: "
echo -n "batch size = 1: "
grep Averager logs/bert-1-tf.log 
echo -n "batch size = 16: "
grep Averager logs/bert-16-tf.log 
echo ""
