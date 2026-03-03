import numpy as np
import os

def main():
    input_file = "StangaI.npz"
    output_file = "StangaI_cleaned.npz" # Meglio salvarlo con un nome diverso per sicurezza

    if not os.path.exists(input_file):
        print(f"Errore: Il file {input_file} non esiste nella cartella corrente.")
        return

    # Carica il file npz
    data = np.load(input_file, allow_pickle=True)
    
    # Creiamo un dizionario per i nuovi dati
    new_data = {}
    
    # Iteriamo su tutte le chiavi presenti nel file (pts1, pts2, undistorted, ecc.)
    for key in data.files:
        val = data[key]
        
        # Se la chiave è pts1 o pts2, togliamo gli ultimi 4 elementi
        if key in ["pts1", "pts2"]:
            if len(val) > 4:
                new_data[key] = val[:-4]
                print(f"-> Rimosse 4 righe da '{key}'. Nuova dimensione: {len(new_data[key])}")
            else:
                new_data[key] = val
                print(f"-> Attenzione: '{key}' ha solo {len(val)} punti, impossibile toglierne 4.")
        else:
            # Per gli altri dati (come il flag 'undistorted'), copiamoli così come sono
            new_data[key] = val

    # Salva il nuovo file
    np.savez(output_file, **new_data)
    print(f"\n✅ Operazione completata. File salvato come: {output_file}")

if __name__ == "__main__":
    main()