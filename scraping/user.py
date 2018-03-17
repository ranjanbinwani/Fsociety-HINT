from bs4 import BeautifulSoup
import requests
users=open("sdf.txt","r")
id=0
try:
	new=open("user_list.txt",'w+')
	for username in users:
		ans=""
		num_tweets=-1
		num_following=-1
		num_followers=-1
		num_likes=-1
		try:
			username=username.strip()
			url = 'https://www.twitter.com/'+username
			r = requests.get(url)
			soup = BeautifulSoup(r.content, 'lxml')

			f = soup.find('li', class_="ProfileNav-item--tweets")
			title = f.find('a')['title']
			
			num_tweets = int(title.split(' ')[0].replace(',',''))

			f = soup.find('li', class_="ProfileNav-item--following")
			title = f.find('a')['title']
			
			num_following = int(title.split(' ')[0].replace(',',''))

			f = soup.find('li', class_="ProfileNav-item--followers")
			title = f.find('a')['title']
			
			num_followers = int(title.split(' ')[0].replace(',',''))

			f = soup.find('li', class_="ProfileNav-item--favorites")
			title = f.find('a')['title']
			
			num_likes = int(title.split(' ')[0].replace(',',''))
		except:
			pass
		id+=1
		# print (id, num_tweets, num_following, num_followers,num_likes)
		ans+=str(id)+' '+ str(username) +' '+str(num_tweets)+' '+str(num_following)+' '+str(num_followers)+' '+str(num_likes)+'\n'
		new.write(ans)
		print(id)
		if(id==500):
			break
		
except:
	pass
new.close()