from configs import Config


args = Config()

with open(args.raw_file, encoding='ascii') as reader:
    data = reader.readline()