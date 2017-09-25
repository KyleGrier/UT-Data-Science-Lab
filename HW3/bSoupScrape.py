import urllib
from bs4 import *

url = 'http://proceedings.mlr.press/v70/'
html = urllib.urlopen(url).read()
soup = BeautifulSoup(html)

tags = soup('a')

for tag in tags:
	# .encode('utf-8') fixes unrecognized character crash
	print tag.get('href', None).encode('utf-8')
