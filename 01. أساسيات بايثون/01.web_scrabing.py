import requests
from bs4 import BeautifulSoup
import re
from datetime import date
from tabulate import tabulate

def get_forcast_data():
    
    url="https://world-weather.info/"
    headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",'cookie':'celsius=1'}

    response= requests.get(url,headers=headers)

    if response.ok:
        soup=BeautifulSoup(response.content,'html.parser')
        resorts=soup.find('div',id='resorts')
        #print(resorts)
        #since the responce will retun the whole test we will RegExr.com to apply the code filter what we want only ==> Cities 
        ## we did so by coping the html code using inspector and try it on the website 
    
        re_cities= r'">([\w\s]+)<\/a><span>'
        cities=re.findall(re_cities,str(resorts))
        #print(cities)

        re_temps= r'<span>(\+\d+|-d+)<span'
        temps=re.findall(re_temps,str(resorts))
        #convert into int 
        temps=[int(temp)for temp in temps]
        #print(temps)
        condtions_tag=resorts.find_all('span',class_='tooltip')
        conditions=[condition.get('title')for condition in condtions_tag]
        #print(conditions)

        # collect the 3 lists in one table 
        data=zip(cities,temps,conditions)

        return data
    
    return  False



def get_forcast_txt():
    data =get_forcast_data()
    if data:
        today=date.today().strftime('%d/%m/%Y')
        with open('output.txt','w') as f:
            f.write('Main Cities Forcast '+ '\n')
            f.write(today +'\n')
            f.write('*'*25 +'\n')
            #table=tabulate(data, headers= ['City', 'Temp.' , 'Condition'], tablefmt='fancy_grid')
            table=tabulate(data, headers=["City", "Temp. C",'Condition'])
            f.write(table)






if __name__ == '__main__' :
    get_forcast_txt()

