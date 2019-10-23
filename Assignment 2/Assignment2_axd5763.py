import os
import copy
import random
import math
import re
import timeit

sub_folders=[]
path=[]      
global_dictionary = {}                                  
dictionary = {}
file_name ={}
global domain
domain='NULL'
path = os.getcwd()+"/20_newsgroups/"

def stop_words_calc():
    pathcheck('stopwords.txt','Stop Words file');
    f=open("stopwords.txt", "r");
    next_line=['\n'];
    if f.mode == 'r':
        stop_words=f.read()       
        for i in next_line:
            stop_words= stop_words.replace(i,' ')
            li = list(stop_words.split(" "))
    text = ([ f' {x} ' for x in li ]);
    text.pop();
    return text;

def load_test_files(sub_folders):               
    global domain                                 
    while (len(sub_folders)):
        random_group = random.randint(0,len(sub_folders)-1)
        random_newsgroup = sub_folders[random_group]
        if len(file_name[random_newsgroup])== 0:
            sub_folders.remove(random_newsgroup)
        else:
            random_file_num = random.randint(0, len(file_name[random_newsgroup])-1)
            random_file = file_name[random_newsgroup][random_file_num]
            file_name[random_newsgroup].remove(random_file)
            domain = random_newsgroup
            text = open_file(path + random_newsgroup + '/'+ random_file);
            return text
    domain = 'NULL'
    return domain

def preprocessing(text,stop_words):                                                                
    symbol_remove = ['<','>','?','.','"',')','(','|','-','#','*','+','\'','&','^','`','~','\t','$','%',"'",'!','/','\\','=',',',':']
    text = text.lower()
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)              ## regex email addresses and removing them
    text = text.replace('\n', ' ')
    for word in email:
        text = text.replace(word,' ')
    for word in stop_words:                         #   Comment this line and the line below to 
        text = text.replace(word,' ');              #   not remove stop words from the dataset
    for word in symbol_remove:
        text = text.replace(word,'')
    text = text.split(' ');
    if '' in text: text.remove('');
    if ' ' in text: text.remove(' ');
    return text
    
def calculate(_file_, dictionary,sub_folder_list):                                   
    probability = 0.0
    predictor_prob = sum(dictionary.values())
    prior_probability=1/len(sub_folder_list)
    for word in _file_:
        likelihood = dictionary.get(word, 0.0000000001)                                  ### change the precision here to increase or decrease accuracy by 2%( range 0.1 to 0.000001)
        probability+= math.log((float(likelihood)*prior_probability)/float(predictor_prob))
    return probability


def open_file(file_location):
    with open(file_location, encoding="latin-1") as datafile:
        text = datafile.read();
    return text


def train_model(sub_folder_list,stop_words):               
    print ("...............................Training the model...............................\n")
    i=1;
    record=[]
    for sub_folder in sub_folder_list:
        print(i/len(sub_folder_list)*100,"% trained");
        i+=1;
        local_dictionary = {};
        sub_folder_path = path + sub_folder;
        files = os.listdir(sub_folder_path);
        training_set =int(len(files)/2)+1;      # half the dataset constituents the training files.
        flag = 0;
        for _file_ in files:
            flag+= 1;
            if flag > training_set:
                break;
            opened_file=open_file(sub_folder_path + '/'+_file_);                            # The dataset needed to be opened in Latin-1 Encoding.
            words=preprocessing(opened_file,stop_words);
            for field in words:
                value = local_dictionary.get(field, 0);
                value_t = global_dictionary.get(field, 0);
                local_dictionary[field]=value+1;                                            # laplacian smoothening
                global_dictionary[field]=value_t+1;
            record.append(_file_);
        files=list(set(files).difference(record));                  ## comment this line to test the entire dataset.
        dictionary[sub_folder] = local_dictionary;
        file_name[sub_folder] = files;

def test_model(sub_folder_list,stop_words):
    print ("\n................................Testing the model................................\n")
    loop = 0
    text = 1
    prediction = 0
    sub_folders = copy.deepcopy(sub_folder_list)
    while (text):
        if(loop>0 and loop%493==0):
            print(int(loop/9966*100),'% tested');
        text = load_test_files(sub_folders);                            ## loadin the test files
        if text =='NULL':
            break;
        loop+= 1;
        confusion_matrix = [];
        words=preprocessing(text,stop_words);
        for sub_folder in sub_folder_list:
            confusion_matrix.append(calculate(words,dictionary[sub_folder],sub_folder_list));
        predicted=max(confusion_matrix);
        if domain == sub_folder_list[confusion_matrix.index(predicted)]:
            prediction +=1;   
    return prediction,loop-1

def pathcheck(path,supporting_file_name):
    if os.path.exists(path):
        print('\n',supporting_file_name,"detected in the current Directory\n");
    else:
        print('\nDataset not present in parent directory. Load dataset and try again.\n Program exiting!!!');
        exit();

def main():
    stop_words= stop_words_calc()
    start = timeit.default_timer();
    pathcheck(path,'20 News Group Dataset');
    sub_folder_list = os.listdir(path);                  ## list of all sub directories
    domain='NULL';
    train_model(sub_folder_list,stop_words);
    prediction,loop= test_model(sub_folder_list,stop_words);
    print ('Prediction accuracy = ',(prediction/loop*100),'%' );
    stop = timeit.default_timer();
    print('Time elapsed: ', int(stop - start),"secs"); 
if __name__== "__main__":
  main()


