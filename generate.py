from omegaconf import OmegaConf

import pandas as pd

from logic.generator import FlowMatchingGenerator

if __name__ == '__main__':

    cfg = OmegaConf.load('config/config.yaml')
    generator = FlowMatchingGenerator(cfg)

    labels = ['1.1.1.100'] * 4
    seqs = [
        'MKMTKSALVTGASRGIGRSIALQLAEEGYNVAVNYAGSKEKAEAVVEEIKAKGVDSFAIQANVADADEVKAMIKEVVSQFGSLDVLVNNAGITRDNLLMRMKEQEWDDVIDTNLKGVFNCIQKATPQMLRQRSGAIINLSSVVGAVGNPGQANYVATKAGVIGLTKSAARELASRGITVNAVAPGFIVSDMTDALSDELKEQMLTQIPLARFGQDTDIANTVAFLASDKAKYITGQTIHVNGGMYM'
    ] * 4

    batch = generator.wrap_batch(seqs, labels)
    generator.generate(batch)

    

