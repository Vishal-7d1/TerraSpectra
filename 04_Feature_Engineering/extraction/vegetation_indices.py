"""
04_Feature_Engineering/extraction/vegetation_indices.py

Narrow-band vegetation and pathological index engineering:
Extracts 10 scientifically documented indices sensitive to:
- General canopy greenness & biomass (NDVI, GNDVI)
- Pre-symptomatic Red-Edge chlorophyll degradation (NDRE, mSR705, MCARI)
- Photosynthetic light-use efficiency & xanthophyll carotenoids (PRI, SIPI, CRI)
- Canopy hydration & cellular water attenuation (WBI)
- Cell senescence & fungal necrosis (PSRI)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from utils.feature_utils import find_nearest_wavelength


class VegetationIndexExtractor:
    """Computes narrow-band physiological indices with zero-division safeguards."""

    def __init__(self, spectral_matrix: np.ndarray, wavelengths: np.ndarray):
        self.spectral_matrix = spectral_matrix
        self.wavelengths = wavelengths
        self.num_samples, self.num_bands = spectral_matrix.shape

    def extract_indices(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Calculates 10 narrow-band vegetation indices."""
        df = pd.DataFrame(index=np.arange(self.num_samples))
        mapping = {}

        # 1. NDVI (800 nm, 680 nm)
        i800, w800 = find_nearest_wavelength(800.0, self.wavelengths)
        i680, w680 = find_nearest_wavelength(680.0, self.wavelengths)
        r800 = self.spectral_matrix[:, i800]
        r680 = self.spectral_matrix[:, i680]
        denom = r800 + r680
        df["VI_NDVI"] = np.where(denom != 0, (r800 - r680) / denom, 0.0)
        mapping["VI_NDVI"] = {"formula": "(R800 - R680)/(R800 + R680)", "wls": [w800, w680]}

        # 2. NDRE (790 nm, 720 nm)
        i790, w790 = find_nearest_wavelength(790.0, self.wavelengths)
        i720, w720 = find_nearest_wavelength(720.0, self.wavelengths)
        r790 = self.spectral_matrix[:, i790]
        r720 = self.spectral_matrix[:, i720]
        denom = r790 + r720
        df["VI_NDRE"] = np.where(denom != 0, (r790 - r720) / denom, 0.0)
        mapping["VI_NDRE"] = {"formula": "(R790 - R720)/(R790 + R720)", "wls": [w790, w720]}

        # 3. PRI (531 nm, 570 nm)
        i531, w531 = find_nearest_wavelength(531.0, self.wavelengths)
        i570, w570 = find_nearest_wavelength(570.0, self.wavelengths)
        r531 = self.spectral_matrix[:, i531]
        r570 = self.spectral_matrix[:, i570]
        denom = r531 + r570
        df["VI_PRI"] = np.where(denom != 0, (r531 - r570) / denom, 0.0)
        mapping["VI_PRI"] = {"formula": "(R531 - R570)/(R531 + R570)", "wls": [w531, w570]}

        # 4. mSR705 (750 nm, 705 nm, 450.8 nm)
        i750, w750 = find_nearest_wavelength(750.0, self.wavelengths)
        i705, w705 = find_nearest_wavelength(705.0, self.wavelengths)
        i450, w450 = find_nearest_wavelength(450.8, self.wavelengths)
        r750 = self.spectral_matrix[:, i750]
        r705 = self.spectral_matrix[:, i705]
        r450 = self.spectral_matrix[:, i450]
        denom = r705 - r450
        df["VI_mSR705"] = np.where(denom != 0, (r750 - r450) / denom, 0.0)
        mapping["VI_mSR705"] = {"formula": "(R750 - R450)/(R705 - R450)", "wls": [w750, w705, w450]}

        # 5. WBI (900 nm, 970 nm)
        i900, w900 = find_nearest_wavelength(900.0, self.wavelengths)
        i970, w970 = find_nearest_wavelength(970.0, self.wavelengths)
        r900 = self.spectral_matrix[:, i900]
        r970 = self.spectral_matrix[:, i970]
        df["VI_WBI"] = np.where(r970 != 0, r900 / r970, 0.0)
        mapping["VI_WBI"] = {"formula": "R900 / R970", "wls": [w900, w970]}

        # 6. CRI (510 nm, 550 nm)
        i510, w510 = find_nearest_wavelength(510.0, self.wavelengths)
        i550, w550 = find_nearest_wavelength(550.0, self.wavelengths)
        r510 = self.spectral_matrix[:, i510]
        r550 = self.spectral_matrix[:, i550]
        t1 = np.where(r510 != 0, 1.0 / r510, 0.0)
        t2 = np.where(r550 != 0, 1.0 / r550, 0.0)
        df["VI_CRI"] = t1 - t2
        mapping["VI_CRI"] = {"formula": "(1/R510) - (1/R550)", "wls": [w510, w550]}

        # 7. PSRI (Plant Senescence: 680 nm, 500 nm, 750 nm)
        i500, w500 = find_nearest_wavelength(500.0, self.wavelengths)
        r500 = self.spectral_matrix[:, i500]
        df["VI_PSRI"] = np.where(r750 != 0, (r680 - r500) / r750, 0.0)
        mapping["VI_PSRI"] = {"formula": "(R680 - R500) / R750", "wls": [w680, w500, w750]}

        # 8. GNDVI (Green NDVI: 800 nm, 550 nm)
        denom = r800 + r550
        df["VI_GNDVI"] = np.where(denom != 0, (r800 - r550) / denom, 0.0)
        mapping["VI_GNDVI"] = {"formula": "(R800 - R550)/(R800 + R550)", "wls": [w800, w550]}

        # 9. MCARI (700 nm, 670 nm, 550 nm)
        i700, w700 = find_nearest_wavelength(700.0, self.wavelengths)
        i670, w670 = find_nearest_wavelength(670.0, self.wavelengths)
        r700 = self.spectral_matrix[:, i700]
        r670 = self.spectral_matrix[:, i670]
        ratio_term = np.where(r670 != 0, r700 / r670, 0.0)
        df["VI_MCARI"] = ((r700 - r670) - 0.2 * (r700 - r550)) * ratio_term
        mapping["VI_MCARI"] = {"formula": "[(R700-R670) - 0.2*(R700-R550)] * (R700/R670)", "wls": [w700, w670, w550]}

        # 10. SIPI (Structure Insensitive Pigment Index: 800 nm, 450.8 nm, 680 nm)
        denom = r800 - r680
        df["VI_SIPI"] = np.where(denom != 0, (r800 - r450) / denom, 0.0)
        mapping["VI_SIPI"] = {"formula": "(R800 - R450)/(R800 - R680)", "wls": [w800, w450, w680]}

        return df, mapping
