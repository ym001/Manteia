import sys,time
import numpy as np

def coss_validation_idx(nb_pass,nb_docs):
  docs_idx = [idx for idx in range(nb_docs)]
  train_idx, test_idx = [], []
  for pli in range(nb_pass):
    test_pli_idx = list(np.random.choice(docs_idx,int(len(docs_idx)/nb_pass) , replace=False))
    train_pli_idx  = [idx for idx in docs_idx if idx not in test_pli_idx]
    train_idx.append(train_pli_idx)
    test_idx.append(test_pli_idx)
  return train_idx, test_idx
  
def progress(count, total):
	bar_len = 60
	filled_len = int(round(bar_len * count / float(total)))

	percents = round(100.0 * count / float(total), 1)
	bar = '=' * filled_len + '-' * (bar_len - filled_len)
	str_print = '[%s] %s%s %s/%s\r' % (bar, percents, '%',count,total)
	if count < total :
		sys.stdout.write(str_print)
	else :
		str_p=''
		for i in range(len(str_print)):
			str_p+=' '
		sys.stdout.write(str_p+'\r')

	sys.stdout.flush()  # As suggested by Rom Ruben
	
def bar_progress(current, total, width=80):
	bar_len = 60
	filled_len = int(round(bar_len * current / float(total)))

	percents = round(100.0 * current / float(total), 1)
	bar = '=' * filled_len + '-' * (bar_len - filled_len)
	str_print = '%s%s [%s] %s/%s\r' % (percents, '%',bar ,current,total)
	if current < total :
		sys.stdout.write(str_print)
	else :
		str_p=''
		for i in range(len(str_print)):
			str_p+=' '
		sys.stdout.write(str_p+'\r')
	#progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
	# Don't use print() as it will print in new line every time.
	#sys.stdout.write("\r" + progress_message)
	sys.stdout.flush()
	
# Function for implementing the loading animation 
def load_animation(load_str,nb_animation): 
  
    # String to be displayed when the application is loading 
    ls_len = len(load_str) 
  
  
    # String for creating the rotating line 
    anicount = 0
      
    # used to keep the track of 
    # the duration of animation 
    counttime = 0        
      
    # pointer for travelling the loading string 
    i = 0                     
    nb_passsage=ls_len*nb_animation
    while (counttime <nb_passsage): 
        time.sleep(0.5)  
        # used to change the animation speed 
        # smaller the value, faster will be the animation 
                              
        # converting the string to list 
        # as string is immutable 
        load_str_list = list(load_str)  
          
        # x->obtaining the ASCII code 
        x = ord(load_str_list[i]) 
          
        # y->for storing altered ASCII code 
        y = 0                             
  
        # if the character is "." or " ", keep it unaltered 
        # switch uppercase to lowercase and vice-versa  
        if x != 32 and x != 46:              
            if x>90: 
                y = x-32
            else: 
                y = x + 32
            load_str_list[i]= chr(y) 
          
        # for storing the resultant string 
        res =''              
        for j in range(ls_len): 
            res = res + load_str_list[j] 
              
        # displaying the resultant string 
        sys.stdout.write("\r"+res) 
        sys.stdout.flush() 
  
        # Assigning loading string 
        # to the resultant string 
        load_str = res 
  
        i =(i + 1)% ls_len 
        counttime = counttime + 1
    print()
