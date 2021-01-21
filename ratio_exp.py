import os
import Brothers
import Evaluation
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

#####################################
brothers=Brothers.Brothers(dataset='cifar10',weights=[0,0,0,1,1,1,1,1,1],gpu_num=4)
brothers.load_brothers()
brothers.load_classifier_ratio(file_name='ratio_0.0')

evaluation=Evaluation.Evaluation(dataset='cifar10',net=brothers.feature_classifier_joint_forward,batch_size=100)
evaluation.load_data_loader()
evaluation.evaluate_accuracy()
# evaluation.evaluate_FGSM(pert=8/255)
# evaluation.evaluate_PGD(pert=8/255,iter=10)
evaluation.evaluate_PGD(pert=8/255,iter=20)

#####################################
brothers=Brothers.Brothers(dataset='cifar10',weights=[0,0,0,1,1,1,1,1,1],gpu_num=4)
brothers.load_brothers()
brothers.load_classifier_ratio(file_name='ratio_0.2')

evaluation=Evaluation.Evaluation(dataset='cifar10',net=brothers.feature_classifier_joint_forward,batch_size=100)
evaluation.load_data_loader()
evaluation.evaluate_accuracy()
# evaluation.evaluate_FGSM(pert=8/255)
# evaluation.evaluate_PGD(pert=8/255,iter=10)
evaluation.evaluate_PGD(pert=8/255,iter=20)

#####################################
brothers=Brothers.Brothers(dataset='cifar10',weights=[0,0,0,1,1,1,1,1,1],gpu_num=4)
brothers.load_brothers()
brothers.load_classifier_ratio(file_name='ratio_0.4')

evaluation=Evaluation.Evaluation(dataset='cifar10',net=brothers.feature_classifier_joint_forward,batch_size=100)
evaluation.load_data_loader()
evaluation.evaluate_accuracy()
# evaluation.evaluate_FGSM(pert=8/255)
# evaluation.evaluate_PGD(pert=8/255,iter=10)
evaluation.evaluate_PGD(pert=8/255,iter=20)

#####################################
brothers=Brothers.Brothers(dataset='cifar10',weights=[0,0,0,1,1,1,1,1,1],gpu_num=4)
brothers.load_brothers()
brothers.load_classifier_ratio(file_name='ratio_0.6')

evaluation=Evaluation.Evaluation(dataset='cifar10',net=brothers.feature_classifier_joint_forward,batch_size=100)
evaluation.load_data_loader()
evaluation.evaluate_accuracy()
# evaluation.evaluate_FGSM(pert=8/255)
# evaluation.evaluate_PGD(pert=8/255,iter=10)
evaluation.evaluate_PGD(pert=8/255,iter=20)

#####################################
brothers=Brothers.Brothers(dataset='cifar10',weights=[0,0,0,1,1,1,1,1,1],gpu_num=4)
brothers.load_brothers()
brothers.load_classifier_ratio(file_name='ratio_0.8')

evaluation=Evaluation.Evaluation(dataset='cifar10',net=brothers.feature_classifier_joint_forward,batch_size=100)
evaluation.load_data_loader()
evaluation.evaluate_accuracy()
# evaluation.evaluate_FGSM(pert=8/255)
# evaluation.evaluate_PGD(pert=8/255,iter=10)
evaluation.evaluate_PGD(pert=8/255,iter=20)

#####################################
brothers=Brothers.Brothers(dataset='cifar10',weights=[0,0,0,1,1,1,1,1,1],gpu_num=4)
brothers.load_brothers()
brothers.load_classifier_ratio(file_name='ratio_1.0')

evaluation=Evaluation.Evaluation(dataset='cifar10',net=brothers.feature_classifier_joint_forward,batch_size=100)
evaluation.load_data_loader()
evaluation.evaluate_accuracy()
# evaluation.evaluate_FGSM(pert=8/255)
# evaluation.evaluate_PGD(pert=8/255,iter=10)
evaluation.evaluate_PGD(pert=8/255,iter=20)


