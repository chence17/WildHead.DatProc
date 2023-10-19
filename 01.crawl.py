'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-06-08 10:15:03
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-18 18:11:59
FilePath: /HoloHead/dataprocess/Web/01.crawl.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os.path as osp
import sys

# from datetime import date
# from icrawler.builtin import FlickrImageCrawler
from icrawler.builtin import BaiduImageCrawler
from icrawler.builtin import BingImageCrawler
from icrawler.builtin import GoogleImageCrawler

FILE_DIR = osp.dirname(osp.abspath(__file__))
WORK_DIR = '/home/tianhao/WebCrawl-Background'
if WORK_DIR not in sys.path:
    sys.path.append(WORK_DIR)
print(WORK_DIR)

keywords = [
    'Nature Background',
    'City Background',
    'Street Background'
]

list_word = sorted(set(keywords))
print(len(list_word), list_word)

max_number = 100

bing_filters = dict(size='large', type='photo')
bing_ex_filters = dict(size='extralarge', type='photo')
google_filters = dict(size='large', type='photo')
pinterest_filters = dict(size='large', type='photo')
flickr_filters = dict(size='large', type='photo')
pexels_filters = dict(size='large', type='photo')

for word in list_word:
    # pinterest crawl
    pinterest_storage = {'root_dir': osp.join(WORK_DIR, 'pinterest', word + ' NP')}
    pinterest_crawler = GoogleImageCrawler(parser_threads=8, downloader_threads=8, storage=pinterest_storage)
    pinterest_crawler.crawl(keyword=f'{word} site:https://www.pinterest.com',
                            filters=pinterest_filters,
                            max_num=max_number)

    # flickr crawl
    flickr_storage = {'root_dir': osp.join(WORK_DIR, 'flickr', word + ' NP')}
    flickr_crawler = GoogleImageCrawler(parser_threads=8, downloader_threads=8, storage=flickr_storage)
    flickr_crawler.crawl(keyword=f'{word} site:https://www.flickr.com', filters=flickr_filters, max_num=max_number)

    # Deprecated, use Google instead
    # Crawl
    # API_KEY: ada8ca2dd883d9608b6312fa57c6f591
    # API_SECRET: 3143654a0a048cad
    # flickr_storage = {'root_dir': osp.join(WORK_DIR, 'flickr', word+' NP')}
    # flickr_crawler = FlickrImageCrawler('ada8ca2dd883d9608b6312fa57c6f591',
    #                                     parser_threads=8,
    #                                     downloader_threads=8,
    #                                     storage=flickr_storage)
    # flickr_crawler.crawl(text=word,
    #                      max_num=max_num,
    #                      min_upload_date=date(2015, 5, 1),
    #                      size_preference=['original', 'large 2048', 'large 1600', 'large'])

    # pexels crawl
    pexels_storage = {'root_dir': osp.join(WORK_DIR, 'pexels', word + ' NP')}
    pexels_crawler = GoogleImageCrawler(parser_threads=8, downloader_threads=8, storage=pexels_storage)
    pexels_crawler.crawl(keyword=f'{word} site:https://www.pexels.com', filters=pexels_filters, max_num=max_number)

    # google crawl
    google_storage = {'root_dir': osp.join(WORK_DIR, 'google', word + ' NP')}
    google_crawler = GoogleImageCrawler(parser_threads=8, downloader_threads=8, storage=google_storage)
    google_crawler.crawl(keyword=word, filters=google_filters, max_num=max_number)

    # bing crawl
    bing_storage = {'root_dir': osp.join(WORK_DIR, 'bing', word + ' NP')}
    bing_crawler = BingImageCrawler(parser_threads=8, downloader_threads=8, storage=bing_storage)
    bing_crawler.crawl(keyword=word, filters=bing_filters, max_num=max_number)

    # bing crawl extra large
    bing_ex_storage = {'root_dir': osp.join(WORK_DIR, 'bingex', word + ' NP')}
    bing_ex_crawler = BingImageCrawler(parser_threads=8, downloader_threads=8, storage=bing_ex_storage)
    bing_ex_crawler.crawl(keyword=word, filters=bing_ex_filters, max_num=max_number)

    # baidu crawl
    baidu_storage = {'root_dir': osp.join(WORK_DIR, 'baidu', word + ' NP')}
    baidu_crawler = BaiduImageCrawler(parser_threads=8, downloader_threads=8, storage=baidu_storage)
    baidu_crawler.crawl(keyword=word, max_num=max_number)
