import bootstrap
from recipebuild_tokenizer import RBTokenizer
from loguru import logger


logger.info(' >> ingr title tag ')
rb_config = bootstrap.recipebuildConfig(path ="/home/jaesung/jaesung/research/recipebuild_retactor/config.json")
rb_tokenizer = RBTokenizer(rb_config)
rb_tokenizer.train()
rb_tokenizer.save()

