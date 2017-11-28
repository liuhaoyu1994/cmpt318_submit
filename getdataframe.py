import pandas as pd
import numpy as np
import cv2
import glob
import sys
import re

class MyImage:
	def __init__(self, img_name):
		self.img = cv2.imread(img_name)
		self.__name = img_name

	def __str__(self):
		return self.__name
#https://stackoverflow.com/questions/44663347/python-opencv-reading-the-image-file-name

datetime_re = re.compile(r'-(\d\d\d\d)(\d\d)(\d\d)(\d\d)(\d\d)')

def get_labels(in_dir):
    output = pd.DataFrame()
    for filename in glob.glob(in_dir+'/'+'*.csv'):
        test = pd.read_csv(filename,skiprows=17,header=None)
        temp = pd.DataFrame({'datetime':test[0],'weather':test[24]})
        output = output.append(temp,ignore_index= True)
    return output

def fill_likelihood(input_row,data):
    weather_list = {'Rain','Cloudy','Mostly Cloudy','Clear','Snow'}
    if input_row['weather'] in weather_list:
        data.set_value(data.index[data['datetime']==input_row['datetime']],[input_row['weather']],1)

def cleanning_data(data):
    data['Rain'] = 0
    data['Cloudy'] = 0
    data['Mostly Cloudy'] = 0
    data['Clear'] = 0
    data['Snow'] = 0
    data.apply(fill_likelihood,axis=1,data = data)
    data.apply(cleanweather,axis=1,data = data)
    data = data[pd.notnull(data['weather'])]
    return data

def cleanweather(input_row,data):
    weather_list = {'Rain','Cloudy','Mostly Cloudy','Clear','Snow'}
    rain_list = {'Rain Showers', 'Moderate Rain', 'Heavy Rain', 'Moderate Rain Showers','Drizzle','Moderate Rain,Drizzle','Rain,Drizzle'}
    snow_list={'Snow Showers', 'Moderate Snow'}
    if input_row['weather'] in weather_list:
        pass
    elif input_row['weather']=='Mainly Clear':
        data.set_value(data.index[data['datetime']==input_row['datetime']],['weather'],'Clear')
    elif input_row['weather'] in rain_list:
        data.set_value(data.index[data['datetime']==input_row['datetime']],['weather'],'Rain')
    elif input_row['weather'] in snow_list:
        data.set_value(data.index[data['datetime']==input_row['datetime']],['weather'],'Snow')
    else:
        data.set_value(data.index[data['datetime']==input_row['datetime']],['weather'],None)
        
def get_datetime(txt):
    match = datetime_re.search(txt)
    if match:
        datetime = match.group(1)+'-'+match.group(2)+'-'+match.group(3)+' '+match.group(4)+':'+match.group(5)
        return datetime
    else:
        return None

def path_to_time(filename):
    datetime = get_datetime(filename)
    return datetime

def is_dark(b,g,r):
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()
    l= [0.2126*x+0.7152*y+0.0722*z for x,y,z in zip(r,g,b)]
    avg_l = sum(l)/len(l)
    return(avg_l<50)

def images_to_pd(in_dir):
    output = pd.DataFrame()
    count = 0
    for filename in glob.glob(in_dir+'/'+'*.jpg'):
        count+=1
        image = MyImage(filename)
        print('working on '+str(count)+'/6991.')
        image2 = image.img[:][0:120]
        BGR = cv2.resize(image2,(30,20),interpolation=cv2.INTER_CUBIC)
        b,g,r = cv2.split(BGR)
        if(is_dark(b,g,r)):
            pass
        else:
            RGB = np.dstack((r,g,b))
            RGB=RGB.flatten()
            datetime = tuple(((path_to_time(str(image))),))
            content = tuple(RGB.tolist())
            d = datetime+content
            temp = pd.DataFrame(data=list(d)).T
            output = output.append(temp,ignore_index= True)
    return output

def join_df(df1,df2):
    df = df2.set_index([0]).join(df1.set_index('datetime'),how = 'inner')
    return df

def main():
    in_dir_csv = sys.argv[1]
    #in_dir_img = sys.argv[2]
    global labels,labels_df
    labels = get_labels(in_dir_csv)
    labels_df = cleanning_data(labels)
    print(labels_df)
    #print('Start reading images:')
    #image_df = images_to_pd(in_dir_img)
    #df = join_df(df1,df2)
    #df.to_csv('sky_no_dark_cubic.csv')
    
    
if __name__ == '__main__':
    main()