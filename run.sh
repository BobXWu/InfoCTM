# /bin/bash
set -e


python crosslingual_topic_model.py --model InfoCTM --dataset ECNews --weight_MI 30.0

python crosslingual_topic_model.py --model InfoCTM --dataset AmazonReview --weight_MI 50.0

python crosslingual_topic_model.py --model InfoCTM --dataset RakutenAmazon --weight_MI 50.0
