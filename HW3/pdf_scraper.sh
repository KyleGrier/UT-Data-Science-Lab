#!/bin/bash

# Scrape pdf URLS
rm /home/sbordovsky/Notes/EE\ 379K/UT-Data-Science-Lab/HW3/pdf_urls.txt
python bSoupScrape.py | grep -v 'supp' | grep --line-buffered '.pdf$' > /home/sbordovsky/Notes/EE\ 379K/UT-Data-Science-Lab/HW3/pdf_urls.txt

# Download pdfs
cd /home/sbordovsky/Notes/EE\ 379K/UT-Data-Science-Lab/HW3/pdfs
cat /home/sbordovsky/Notes/EE\ 379K/UT-Data-Science-Lab/HW3/pdf_urls.txt | while read line
do
	wget $line
done
