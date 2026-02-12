import astropy.units as u
from astroquery.nist import Nist

# Configuration
elements_mapping = {
    "Al I": ("Al", 1),
    "Ca I": ("Ca", 1),
    "Ca II": ("Ca", 2),
    "Ti I": ("Ti", 1),
    "Ti II": ("Ti", 2),
    "Cr I": ("Cr", 1),
    "Mn I": ("Mn", 1),
    "Fe I": ("Fe", 1),
    "Co I": ("Co", 1),
    "Ni I": ("Ni", 1),
    
}
output_filename = "New_LineList_Entries.txt"

with open(output_filename, "w") as f:
    # Header remains the same
    f.write("element\tsp_num\tritz_wl_vac(A)\tAki(s^-1)\tfik\tAcc\tEi(cm-1)\tEk(cm-1)\tType\n")

    for el_name, (sym, sp) in elements_mapping.items():
        print(f"Processing {el_name}...")
        res = Nist.query(3300 * u.AA, 8000 * u.AA, linename=el_name, energy_level_unit='cm-1')

        row_count = 0
        for row in res:
            if row['fik'] == '--' or row['Aki'] == '--':
                continue
            
            def clean_and_quote(val):
                # This function strips NIST formatting and adds the mandatory double quotes
                v = str(val).strip('[]*? ').replace(' ', '')
                return f'"{v}"'

            fik_raw = str(row['fik']).strip('[]*? ')
            try:
                if float(fik_raw) < 0.015:
                    continue
            except ValueError:
                continue

            # Split energy levels
            energy_str = str(row['Ei           Ek'])
            ei, ek = '"0.00"', '""'
            if '-' in energy_str:
                parts = energy_str.split('-')
                ei = clean_and_quote(parts[0])
                ek = clean_and_quote(parts[1])

            # Prepare columns (Element and sp_num do NOT get quotes in the original file)
            wl = clean_and_quote(row['Ritz'])
            aki = clean_and_quote(row['Aki'])
            fik = clean_and_quote(row['fik'])
            acc = clean_and_quote(row['Acc.'])
            line_type = clean_and_quote(row['Type']) if row['Type'] != '--' else '""'

            # 9 columns total, tab separated
            f.write(f"{sym}\t{sp}\t{wl}\t{aki}\t{fik}\t{acc}\t{ei}\t{ek}\t{line_type}\n")
            row_count += 1
            
        print(f"  Added {row_count} lines.")