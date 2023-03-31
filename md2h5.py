
# import codecs, markdown
# import argparse
# def parse_args():
#     parser = argparse.ArgumentParser(description='Convert .md to .html')
#     parser.add_argument('-i', dest='input_file',
#                         help='path of input .md file',type=str)
#     parser.add_argument('-o', dest='output_file', help='path of output .html file',type=str)
#     # parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
#     # parser.add_argument('--manualSeed', type=int, help='manual seed')
#     args = parser.parse_args()
#     return args
# args = parse_args()
# input_file = codecs.open(args.input_file, mode="r", encoding="utf-8")
# text = input_file.read()
# html = markdown.markdown(text)
# output_file = codecs.open(args.output_file, mode="w", encoding="utf-8")
# output_file.write(html)
import pandoc
