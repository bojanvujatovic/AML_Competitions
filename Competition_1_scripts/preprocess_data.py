import gzip

# defining paths
train_file_path = r"/media/bojan/C662068462067A07/UCL/Modules/Applied Machine Learning/Competition_1/Data/train"
train_file_output_path = r"/media/bojan/C662068462067A07/UCL/Modules/Applied Machine Learning/Competition_1/Data/train_preprocessed2"

count = 0

train_file_output = open(train_file_output_path, 'w')

with open(train_file_path, 'r') as train_file:  
    for line in train_file:
        # printing where we are currently
        count += 1
        if count % 1000000 == 0:
            print "Curretntly row:", count
        
        # skip header
        if count == 1:
            continue
        
        # extract all features into list        
        features = line.split(',')
        
        # output label
        if features[1] == '0':
            train_file_output.write("-1 | ")
        else:
            train_file_output.write("1 | ")
        
        # output day and hour
        train_file_output.write("1_" + features[2][4:6] + " ")
        train_file_output.write("2_" + features[2][6:] + " ")
        
        # output the rest of features
        for i in range(3, len(features)):
            train_file_output.write(str(i) + "_" + features[i] + " ")
        
train_file_output.close()

    
