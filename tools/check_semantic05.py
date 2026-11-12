"""Compare 0.7 with 0.5 and optionally build from the exact winning semantic07 ZIP."""
import argparse
import check_semantic_contribution as check

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',action='store_true')
    args=parser.parse_args()
    check.BASE=check.ROOT/'dist/energytool-unified-v1-hybrid-semantic07.zip'
    check.SHA='4d1b307e7a786c29d74824c453a3e21249841cbb04474e44e2e5c4bbe2579f8c'
    check.OUTPUT=check.ROOT/'dist/energytool-unified-v1-hybrid-semantic05.zip'
    check.main(args.build,baseline=0.7,candidate=0.5)
