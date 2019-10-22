import os
import copy
import random
import math
import re
import timeit

sub_folders=[]
file_name = []
path=[]
stop_words = []      
global group
group='NULL'
cwd = os.getcwd()
path = cwd+"/20_newsgroups/"
total_dic = {}                                  
dictionary = {}
file_name ={}


def stop_words_calc():
    stop_words=[]
    f=open("stopwords.txt", "r");
    next_line=['\n'];
    if f.mode == 'r':
        stop_words=f.read()       
        for i in next_line:
            stop_words= stop_words.replace(i,' ')
            li = list(stop_words.split(" "))
    data = ([ f' {x} ' for x in li ]);
    data.pop()
    return data;


def text_preprocessing(text):                                                                   ### cleaned up
    symbol_remove = ['<','>','?','.','"',')','(','|','-','#','*','+','\'','&','^','`','~','\t','$','%',"'",'!','/','\\','=',',',':']
    text = text.replace('\n', ' ')
    text = text.lower()
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)              ## regex email addresses and removing them
    for word in email:
        text = text.replace(word,' ')
    for word in stop_words:
        text = text.replace(word,' ');
    for word in symbol_remove:
        text = text.replace(word,'')
    return text

def load_test_files(sub_folders):               ## change this to do just last 50%
    global group                                 
    while (len(sub_folders)):
        r_fo = random.randint(0,len(sub_folders)-1)
        sub_foldern = sub_folders[r_fo]
        if len(file_name[sub_foldern])== 0:
            sub_folders.remove(sub_foldern)
        else:
            r_fi = random.randint(0, len(file_name[sub_foldern])-1)
            fil = file_name[sub_foldern][r_fi]
            file_name[sub_foldern].remove(fil)
            group = sub_foldern
            data = open_file(path + sub_foldern + '/'+ fil);
            return data
    group = 'NULL'
    return 'NULL'
    
def get_probability(fields, dic):                                   ## understand and change this
    sum_ = sum(dic.values())
    probability = 0.0
    for f in fields:
        value = dic.get(f, 0.0) + 0.0001
        probability = probability + math.log(float(value)/float(sum_))
    return probability


def open_file(file_location):
    with open(file_location, encoding="latin-1") as datafile:
        data = datafile.read();
    return data


def train_model(sub_folderlist):
    print ("...............................Training the model...............................\n")
    for fo in sub_folderlist:
        # print('.')
        dic = {}
        sub_folder = path + fo
        files = os.listdir(sub_folder)
        training_set =int(len(files)/2)+1      ## half the dataset constituents the training files.
        flag = 0
        for fi in files:
            flag+= 1
            if flag > training_set:
                break
            opened_file=open_file(sub_folder + '/'+fi);
            data = text_preprocessing(opened_file);
            fields = data.split(' ');
            if '' in fields: fields.remove('');
            if ' ' in fields: fields.remove(' ');
            for field in fields:
                # if field == ' ' or field == '':
                #     continue
                value = dic.get(field, 0)
                value_t = total_dic.get(field, 0)
                dic[field]=value+1;                     ## laplacian smoothening by default
                total_dic[field]=value_t+1
            files.remove(fi)
        file_name[fo] = files
        dictionary[fo] = dic
    print (len(total_dic), 'unique words were discovered during training.\n');

def test_model(sub_folderlist):
    print ("\n...............................Testing the model...............................")
    data = 1
    sub_folders = copy.deepcopy(sub_folderlist)
    iteration = 0
    success = 0
    while (data):
        data = load_test_files(sub_folders)                            ## loadin the test files
        iteration = iteration + 1
        if data =='NULL':
            break
        data = text_preprocessing(data)
        fields = data.split(' ')
        if '' in fields: fields.remove('')
        if ' ' in fields: fields.remove(' ')
        probabilities = []
        for c in sub_folderlist:
            probabilities.append(get_probability(fields,dictionary[c]))
        if group == sub_folderlist[probabilities.index(max(probabilities))]:
            success = success + 1   
    return success,iteration

def main():
    start = timeit.default_timer()
    sub_folderlist = os.listdir(path)                  ## list of all sub directories
    stop_words = stop_words_calc()
    group='NULL'
    train_model(sub_folderlist);
    success,iteration= test_model(sub_folderlist);
    print ('Success rate = %.1f'% (float(success)/float(iteration - 1)*100))
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
if __name__== "__main__":
  main()


