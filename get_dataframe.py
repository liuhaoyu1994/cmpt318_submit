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
#idea from https://stackoverflow.com/questions/44663347/python-opencv-reading-the-image-file-name


# Csv read/cleaning functions: 
#==============================================================================
#weather_list = {'Rain','Cloudy','Clear','Snow'}
weather_list = {'Rain','Cloudy','Clear'}
rain_list = {'Rain Showers', 'Moderate Rain', 'Heavy Rain', 'Moderate Rain Showers','Drizzle','Moderate Rain,Drizzle','Rain,Drizzle'}
#snow_list={'Snow Showers', 'Moderate Snow'}

#==============================================================================

def get_labels(in_dir):
    '''
    Read csv files in given directory and return a dataframe with datetime and weather label for each time point
    '''
    output = pd.DataFrame()
    for filename in glob.glob(in_dir+'/'+'*.csv'):
        test = pd.read_csv(filename,skiprows=17,header=None)
        temp = pd.DataFrame({'datetime':test[0],'weather':test[24]})
        output = output.append(temp,ignore_index= True)
    return output


def cleanning_data(data):
    '''
    Apply relabel(), fill_likelihood() and generate_likelihood() for input dataframe. 
    Drop rows with no likelihood assigned. Return transformed dataframe.
    '''
    for i in weather_list:
        data[i] = 0
    data.apply(relabel,axis=1,data = data)
    data['weather_re']=data['weather']
    data.apply(fill_likelihood,axis=1,data = data)
    data.apply(generate_likelihood,axis=1,data = data)
    #data = data.drop(data[data.Rain+data.Cloudy+data.Clear+data.Snow==0].index)
    data = data.drop(data[data.Rain+data.Cloudy+data.Clear==0].index)
    return data


def relabel(input_row,data):
    '''
    Classify labels into 3 classes. Set uncommon weather labels to None.
    '''
    row_index = data.index[data['datetime']==input_row['datetime']].tolist()
    if input_row['weather'] in weather_list:
        pass
    elif input_row['weather']=='Mainly Clear':
        data.loc[row_index[0],['weather']]='Clear'
    elif input_row['weather']=='Mostly Cloudy':
        data.loc[row_index[0],['weather']]='Cloudy'
    elif input_row['weather'] in rain_list:
        data.loc[row_index[0],['weather']]='Rain'
    #elif input_row['weather'] in snow_list:
    #    data.loc[row_index[0],['weather']]='Snow'
    else:
        data.loc[row_index[0],['weather']]=None


def fill_likelihood(input_row,data):
    '''
    Assign weather likelihood for labeled rows.
    '''
    row_index = data.index[data['datetime']==input_row['datetime']].tolist()
    if input_row['weather'] in weather_list:
        data.loc[row_index[0],[input_row['weather']]] = 1


def generate_likelihood(input_row,data):
    '''
    Assign weather likelihood for unlabeled rows.
    '''
    ignore = {0,1,data.index.size-1,data.index.size-2}
    row_index = data.index[data['datetime']==input_row['datetime']].tolist()
    if pd.isnull(input_row['weather']) and row_index[0] not in ignore:
        max_likelihood=0
        label_re = None
        for i in weather_list:
            value = 0
            for neighbour in {row_index[0]-2,row_index[0]-1,row_index[0]+1,row_index[0]+2}:
                if pd.notnull(data.loc[neighbour,'weather']):
                    value += data.loc[neighbour,i]*0.5/abs(neighbour-row_index[0])
            data.loc[row_index[0],[i]] = value
            if max_likelihood < value:
                max_likelihood = value
                label_re = i    
        data.loc[row_index[0],['weather_re']]=label_re


# File name to datetime functions:
#==============================================================================
datetime_re = re.compile(r'-(\d\d\d\d)(\d\d)(\d\d)(\d\d)(\d\d)')
        
def get_datetime(txt):
    '''
    Match the regular expression to find datetime in filename. Return datetime with same format as it in csv.
    '''
    match = datetime_re.search(txt)
    if match:
        datetime = match.group(1)+'-'+match.group(2)+'-'+match.group(3)+' '+match.group(4)+':'+match.group(5)
        return datetime
    else:
        return None

def path_to_time(filename):
    '''
    Read filename and return a datetime with same format as it in csv.
    '''
    datetime = get_datetime(filename)
    return datetime


# Image read/cleaning functions: 
#==============================================================================
def images_to_pd(in_dir):
    '''
    Read all images in given directory. 
    Return a dataframe contains datetime, average luminance, number of non zero count in edges and RGB values of images with average luminance greater than 50.
    '''
    output = pd.DataFrame()
    count = 0
    for filename in glob.glob(in_dir+'/'+'*.jpg'):
        count+=1
        image = MyImage(filename)
        print('working on '+str(count)+'/6991.')
        sky = image.img[:][0:120]
        #BGR = image2
        edges = cv2.Canny(sky,20,50)
        b,g,r = cv2.split(sky)
        avg_l = get_avg_l(b,g,r)
        BGR = cv2.resize(sky,(128,96),interpolation=cv2.INTER_NEAREST)
        b_re,g_re,r_re = cv2.split(BGR)
        if(avg_l<50):
            pass
        else:
            RGB = np.dstack((r_re,g_re,b_re))
            RGB=RGB.flatten()
            datetime = [path_to_time(str(image))]
            content = RGB.tolist()
            avg_l = [avg_l]
            edge_count = [str(np.count_nonzero(edges))]
            d = datetime+avg_l+edge_count+content
            temp = pd.DataFrame(data=d).T
            output = output.append(temp,ignore_index= True)
    output.rename(columns={'Unnamed: 0':'datetime'}, inplace=True)
    output.rename(columns={1:'avg_l'}, inplace=True)
    output.rename(columns={2:'edge_count'}, inplace=True)  
    return output


def get_avg_l(b,g,r):
    '''
    Return a average luminace value for given RGB values.
    '''
    l= [0.2126*x+0.7152*y+0.0722*z for x,y,z in zip(r.flatten(),g.flatten(),b.flatten())]
    avg_l = sum(l)/len(l)
    return(avg_l)


#==============================================================================

def join_df(df1,df2):
    '''
    Joining input dataframes on 'datetime'.
    '''
    df = df2.set_index('datetime').join(df1.set_index([0]),how = 'inner')
    return df

#==============================================================================
    
def main():
    in_dir_csv = sys.argv[1]
    in_dir_img = sys.argv[2]
    labels = get_labels(in_dir_csv)
    labels_df = cleanning_data(labels)
    print('Start reading images:')
    images_df = images_to_pd(in_dir_img)
    df = join_df(images_df,labels_df)
    df.to_csv('image_labeled.csv')
    
    
if __name__ == '__main__':
    main()