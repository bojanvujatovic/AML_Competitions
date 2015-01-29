clear

echo "Preprocessing dataset..."
#python preprocess_data.py 

echo "Running traning..."
vw --random_seed 42 --data "/media/bojan/C662068462067A07/UCL/Modules/Applied Machine Learning/Competition_1/Data/train_preprocessed2" --passes 5 --cache --loss_function logistic -b 22


