def emailFiles():
    import glob
    import os
  
    files = glob.glob(os.getcwd()+"/emails_by_address/from*.txt")
    return files

def addressExtract(email_file):
    address = email_file.rsplit('\\',1)[1]
    address = address.split('_',1)[1]
    
    return address


def processEmail(file_in, address, poi_flag, subjects, bodies, pretty = False, small = True):
    import codecs
    import json
    
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "a") as fo:
        for i in range(len(subjects)):
            el = {}
            el['address'] = address
            el['poi_flag'] = poi_flag
            el['subject'] = subjects[i]
            el['body'] = bodies[i]
            
            if pretty:
                fo.write(json.dumps(el, indent=2)+"\n")
            else:
                fo.write(json.dumps(el) + "\n")
                
        if small:
            data = []
        
    return data


def emailParser(email_list):
    import os    
    from parse_out_email_subject import parseOutSubject
    from parse_out_email_subject import parseOutBody
        
    temp_counter = 0
    subject_data = []
    body_data = []
    
    for path in email_list:
        path = path.split('/',1)[1]        
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        #temp_counter += 1
        if temp_counter < 200:
            path = os.path.join('..', path[:-1])
            try:            
                email = open(path, "r")
            
                ### use parseOutText to extract the text from the opened email
                stemmed_subj = parseOutSubject(email)
                stemmed_body = parseOutBody(email)
                    
                subject_data.append(stemmed_subj)
                body_data.append(stemmed_body)
                
                email.close()
            except:
                pass
    return subject_data, body_data

if __name__ == "__main__":
    
    from progressbar import ProgressBar 
    
    
    email_file_list = emailFiles()
    from poi_email_addresses import poiEmails
    
    poi_emails = poiEmails()
    
    file_in = 'processed_emails'
    
    pbar = ProgressBar()
    for email_file in pbar(email_file_list): 
        email_list = open(email_file, "r")
        address = addressExtract(email_file)
        
        poi_flag = 0
        if address[:-4] in poi_emails:
            poi_flag = 1
        
        subjects, bodies = emailParser(email_list)
        processEmail(file_in, address, poi_flag, subjects, bodies, pretty = False, small = True )
        
        email_list.close()
    
    json_file = 'processed_emails.json'
    from mongo_related import *
    
    import_to_db(json_file)
    
    
#import pickle
#
#pickle.dump( subject_data, open("subject_word_data.pkl", "w") )
#pickle.dump( poi_data, open("subject_email_authors.pkl", "w") )
#pickle.dump( subject_data, open("subject_word_data.pkl", "w") )
#
#
# 