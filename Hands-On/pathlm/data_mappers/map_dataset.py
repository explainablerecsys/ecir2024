import argparse

from pathlm.data_mappers.mapper_cafe import MapperCAFE
from pathlm.data_mappers.mapper_kgat import MapperKGAT
from pathlm.data_mappers.mapper_pgpr import MapperPGPR

PGPR = 'pgpr'
CAFE = 'cafe'
UCPR = 'ucpr'
KGAT = 'kgat'
CKE = 'cke'
CFKG = 'cfkg'
BPRMF = 'bprmf'
NFM = 'nfm'
FM = 'fm'
TRANSE = 'transe'
PLM = 'plm'
PEARLM = 'pearlm'
SUPPORTED_MODELS = [PGPR, CAFE, UCPR, KGAT, CKE, CFKG, BPRMF, NFM, FM, TRANSE, PLM, PEARLM]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ml1m', help='One of {ML1M, LFM1M}')
    parser.add_argument('--model', type=str, default=PGPR, help='')
    parser.add_argument('--train_size', type=float, default=0.6, help='size of the train set expressed in 0.x')
    parser.add_argument('--valid_size', type=float, default=0.2, help='size of the valid set expressed in 0.x')
    args = parser.parse_args()

    if args.model == PGPR:
        MapperPGPR(args)
    elif args.model == CAFE:
        MapperCAFE(args)
    elif args.model == UCPR:
        #MapperUCPR(args)
        pass
    elif args.model == KGAT:
        MapperKGAT(args)
    elif args.model == CKE:
        # Kgat mapper holds correct also for cke
        MapperKGAT(args)
    elif args.model == CFKG:
        # Kgat mapper holds correct also for cfkg
        MapperKGAT(args)
    elif args.model == BPRMF:
        # Kgat mapper holds correct also for bprmf
        MapperKGAT(args)        
    elif args.model == NFM:
        # Kgat mapper holds correct also for nfm
        MapperKGAT(args)        
    elif args.model == FM:
        # Kgat mapper holds correct also for fm
        MapperKGAT(args)  
    elif args.model == TRANSE or args.model == PLM or args.model == PEARLM:
        print('The model selected saves in preprocessed/mapping/ directory')
        args.model = 'mapping'
        MapperPGPR(args)


if __name__ == '__main__':
    main()
