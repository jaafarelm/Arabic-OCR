"""
label_map.py — Maps integer class IDs to human-readable letter names.

Generated from the dataset's folder names. NOTE: class 24 in the shipped
label_mapping.csv was the junk ".ipynb_checkpoints" folder, so it is set to
"Unknown" here rather than a fake letter name. All other entries were spot-
checked against rendered sample images and match.

Usage:
    from label_map import name_for
    name_for(45)  # -> "Ha_Start"
"""

LABEL_NAMES = {
    0: "Sin_End", 1: "Gen_Middle", 2: "Lam_Alf_Mad_End", 3: "Yaa_End",
    4: "Mem_Start", 5: "Saad_Start", 6: "Taa_Start", 7: "Dal_Isolated",
    8: "Gem_Isolated", 9: "Six", 10: "Non_End", 11: "Alf_Hamza_Under_Isolated",
    12: "Sin_Middle", 13: "Gem_Start", 14: "Ain_Middle", 15: "Saad_Middle",
    16: "Haa_End", 17: "Taa_End", 18: "Sin_Isolated", 19: "Zah_Isolated",
    20: "Yaa_Middle", 21: "Three", 22: "Baa_Start", 23: "Yaa_Dot_End",
    24: "Unknown", 25: "Alf_Hamza_Above_Isolated", 26: "Non_Middle",
    27: "Lam_Alf_Hamza_End", 28: "Khaa_End", 29: "Alf_Hamza_Under_End",
    30: "Shen_Start", 31: "Gem_End", 32: "Eight", 33: "Shen_Middle",
    34: "Daad_End", 35: "Alf_Hamza_Above_End", 36: "Non_Start", 37: "Thaa_Start",
    38: "Zah_Middle", 39: "Alf_End", 40: "Zin_End", 41: "Kaf_Middle",
    42: "Raa_Isolated", 43: "Taa_Isolated", 44: "Faa_Middle", 45: "Ha_Start",
    46: "Alf_Isolated", 47: "Khaa_Isolated", 48: "Ha_Isolated",
    49: "Yaa_Dot_Start", 50: "Qaf_Middle", 51: "Saad_End", 52: "Hamza_Isolated",
    53: "Waw_End", 54: "Non_Isolated", 55: "Daad_Start", 56: "Qaf_Start",
    57: "Saad_Isolated", 58: "Tah_Middle", 59: "Ha_End", 60: "Four",
    61: "Seven", 62: "Lam_Alf_Mad_Isolated", 63: "Gen_Isolated", 64: "Khaa_Start",
    65: "Thaa_Isolated", 66: "Kaf_End", 67: "Five", 68: "Lam_Isolated",
    69: "Haa_Start", 70: "Baa_Middle", 71: "Ain_Isolated", 72: "Kaf_Isolated",
    73: "Ain_End", 74: "Zal_Isolated", 75: "Qaf_Isolated",
    76: "Waw_Hamza_Isolated", 77: "Lam_Alf_End", 78: "Sin_Start", 79: "Dal_End",
    80: "Zal_End", 81: "Kaf_Start", 82: "Waw_Isolated", 83: "Daad_Isolated",
    84: "Lam_Middle", 85: "Raa_End", 86: "Mem_End", 87: "Daad_Middle",
    88: "Tah_Isolated", 89: "Baa_Isolated", 90: "Shen_End", 91: "Two",
    92: "Thaa_End", 93: "Lam_Start", 94: "Lam_Alf_Hamza_Isolated",
    95: "Shen_Isolated", 96: "Lam_End", 97: "Ha_Middle", 98: "Faa_End",
    99: "Lam_Alf_Isolated", 100: "Yaa_Isolated", 101: "Mem_Isolated",
    102: "Taa_Middle", 103: "Zero", 104: "Zin_Isolated", 105: "Qaf_End",
    106: "Yaa_Dot_Isolated", 107: "Nine", 108: "Thaa_Middle", 109: "Ain_Start",
    110: "One", 111: "Faa_Start", 112: "Baa_End", 113: "Gen_End",
    114: "Faa_Isolated",
}


def name_for(class_id):
    """Return the human-readable name for a class ID (fallback: 'Class N')."""
    return LABEL_NAMES.get(class_id, f"Class {class_id}")
