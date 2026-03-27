"""
Transport Mechanism Data Collector

Collects data from various sources for BBB transport mechanism prediction:
- ChEMBL: PAMPA, SLC, ABC transporter data
- DrugBank: CNS drug classification
- Metrabase: Transporter interactions
- Literature datasets: BBB permeability assays

Author: BBB Prediction Project
Reference: Cornelissen et al., J. Med. Chem. 2022
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransportDataCollector:
    """
    Collects and curates transport mechanism data from multiple sources.

    Datasets:
    1. PAMPA: Passive diffusion assay data (ChEMBL target: PAMPA-BBB)
    2. Influx: SLC transporter substrates (ChEMBL targets: SLC22, SLCO)
    3. Efflux: ABC transporter substrates (ChEMBL targets: ABCB1, ABCG2, ABCC1-4)
    4. CNS: Central Nervous System drugs (DrugBank)
    5. BBB: Endothelial BBB permeability (B3DB - already available)
    """

    def __init__(self, output_dir: str = "data/transport_mechanisms"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ChEMBL API endpoint
        self.chembl_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.chembl_url_new = "https://www.ebi.ac.uk/chembl/api/data"

        # Create subdirectories
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "curated").mkdir(exist_ok=True)

    def fetch_chembl_data(
        self,
        target_type: str,
        assay_keywords: List[str],
        max_records: int = 5000,
    ) -> pd.DataFrame:
        """
        Fetch data from ChEMBL database.

        Args:
            target_type: Type of target (e.g., 'SLC22', 'ABC', 'PAMPA')
            assay_keywords: Keywords to search in assay descriptions
            max_records: Maximum number of records to fetch

        Returns:
            DataFrame with compound bioactivity data
        """
        logger.info(f"Fetching ChEMBL data for {target_type}...")

        all_data = []

        # Search for assays
        for keyword in assay_keywords:
            try:
                # Search assays
                search_url = f"{self.chembl_url_new}/assay.json"
                params = {
                    "json": "true",
                    "search_query": keyword,
                    "limit": 1000,
                }

                response = requests.get(search_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if "assays" not in data:
                    logger.warning(f"No assays found for keyword: {keyword}")
                    continue

                assays = data["assays"]
                logger.info(f"Found {len(assays)} assays for '{keyword}'")

                # Get activities for each assay
                for assay in assays[:50]:  # Limit to prevent timeout
                    assay_chembl_id = assay["assay_chembl_id"]

                    # Get activities
                    activity_url = f"{self.chembl_url_new}/activity.json"
                    activity_params = {
                        "json": "true",
                        "assay_chembl_id": assay_chembl_id,
                        "limit": 100,
                    }

                    try:
                        act_response = requests.get(
                            activity_url, params=activity_params, timeout=30
                        )
                        act_response.raise_for_status()
                        act_data = act_response.json()

                        if "activities" in act_data:
                            for activity in act_data["activities"]:
                                all_data.append(
                                    {
                                        "target_type": target_type,
                                        "keyword": keyword,
                                        "assay_id": assay_chembl_id,
                                        "molecule_chembl_id": activity.get(
                                            "molecule_chembl_id"
                                        ),
                                        "smiles": activity.get("canonical_smiles"),
                                        "standard_type": activity.get("standard_type"),
                                        "standard_value": activity.get("standard_value"),
                                        "standard_units": activity.get("standard_units"),
                                        "activity_comment": activity.get(
                                            "activity_comment"
                                        ),
                                        "pchembl_value": activity.get("pchembl_value"),
                                    }
                                )

                        # Rate limiting
                        time.sleep(0.1)

                    except Exception as e:
                        logger.warning(f"Error fetching activities: {e}")
                        continue

                # Rate limiting between searches
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching data for keyword '{keyword}': {e}")
                continue

        if not all_data:
            logger.warning(f"No data collected for {target_type}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Save raw data
        output_path = self.output_dir / "raw" / f"{target_type}_raw.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")

        return df

    def curate_pampa_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Curate PAMPA (passive diffusion) data.

        Classification criteria (from Cornelissen et al.):
        - Permeable: Pe > 4 × 10⁻⁶ cm/s
        - Impermeable: Pe < 2 × 10⁻⁶ cm/s
        - Intermediate: excluded
        """
        logger.info("Curating PAMPA data...")

        if df.empty:
            logger.warning("No PAMPA data to curate")
            return pd.DataFrame()

        curated = []

        for _, row in df.iterrows():
            try:
                # Extract permeability value
                if pd.notna(row.get("pchembl_value")):
                    pe = float(row["pchembl_value"])
                elif pd.notna(row.get("standard_value")):
                    pe = float(row["standard_value"])
                else:
                    continue

                # Classify based on permeability
                if pe > 4.0:  # Permeable (Pe > 4 × 10⁻⁶ cm/s)
                    label = 1  # Permeable
                elif pe < 2.0:  # Impermeable (Pe < 2 × 10⁻⁶ cm/s)
                    label = 0  # Impermeable
                else:
                    continue  # Exclude intermediate

                curated.append(
                    {
                        "smiles": row.get("smiles"),
                        "permeability": pe,
                        "label": label,
                        "assay_id": row.get("assay_id"),
                    }
                )

            except (ValueError, TypeError):
                continue

        curated_df = pd.DataFrame(curated)

        if not curated_df.empty:
            # Remove duplicates and invalid SMILES
            curated_df = curated_df.dropna(subset=["smiles"])
            curated_df = curated_df.drop_duplicates(subset=["smiles"])

            # Save curated data
            output_path = self.output_dir / "curated" / "pampa_curated.csv"
            curated_df.to_csv(output_path, index=False)
            logger.info(
                f"Curated {len(curated_df)} PAMPA compounds "
                f"({curated_df['label'].sum()} permeable)"
            )

        return curated_df

    def curate_influx_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Curate Influx (SLC transporter) data.

        Classification criteria (from Cornelissen et al.):
        - Substrate: Active uptake / substrate
        - Nonsubstrate: Inactive uptake / nonsubstrate / inhibitor
        """
        logger.info("Curating Influx (SLC) data...")

        if df.empty:
            logger.warning("No Influx data to curate")
            return pd.DataFrame()

        curated = []

        for _, row in df.iterrows():
            smiles = row.get("smiles")
            if not smiles or pd.isna(smiles):
                continue

            # Classify based on activity comments and values
            comment = str(row.get("activity_comment", "")).lower()
            pchembl = row.get("pchembl_value")

            label = None

            # Check for substrate indicators
            if any(
                word in comment
                for word in [
                    "substrate",
                    "uptake",
                    "transport",
                    "permeable",
                    "active",
                ]
            ):
                label = 1  # Substrate
            elif any(
                word in comment
                for word in ["nonsubstrate", "inhibitor", "inactive", "non-uptake"]
            ):
                label = 0  # Nonsubstrate
            elif pd.notna(pchembl):
                # Use activity threshold
                if pchembl > 5.0:  # Active (pChEMBL > 5 = < 10 μM)
                    label = 1
                else:
                    label = 0

            if label is not None:
                curated.append(
                    {
                        "smiles": smiles,
                        "label": label,
                        "target_type": row.get("target_type"),
                        "pchembl_value": pchembl,
                        "assay_id": row.get("assay_id"),
                    }
                )

        curated_df = pd.DataFrame(curated)

        if not curated_df.empty:
            # Remove duplicates
            curated_df = curated_df.dropna(subset=["smiles"])
            curated_df = curated_df.drop_duplicates(subset=["smiles"])

            # Save curated data
            output_path = self.output_dir / "curated" / "influx_curated.csv"
            curated_df.to_csv(output_path, index=False)
            logger.info(
                f"Curated {len(curated_df)} Influx compounds "
                f"({curated_df['label'].sum()} substrates)"
            )

        return curated_df

    def curate_efflux_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Curate Efflux (ABC transporter) data.

        Classification criteria (from Cornelissen et al.):
        - Substrate: Efflux ratio (ER) ≥ 5
        - Nonsubstrate: ER ≤ 1
        - Intermediate: excluded
        """
        logger.info("Curating Efflux (ABC) data...")

        if df.empty:
            logger.warning("No Efflux data to curate")
            return pd.DataFrame()

        curated = []

        for _, row in df.iterrows():
            smiles = row.get("smiles")
            if not smiles or pd.isna(smiles):
                continue

            # Try to extract efflux ratio
            comment = str(row.get("activity_comment", "")).lower()
            standard_value = row.get("standard_value")
            pchembl = row.get("pchembl_value")

            label = None

            # Check for efflux ratio in comment
            if "er" in comment or "efflux ratio" in comment:
                # Try to extract numeric value
                import re

                er_match = re.search(r"er\s*:?\s*(\d+\.?\d*)", comment)
                if er_match:
                    er = float(er_match.group(1))
                    if er >= 5:
                        label = 1  # Substrate
                    elif er <= 1:
                        label = 0  # Nonsubstrate
                    else:
                        continue  # Exclude intermediate

            # Alternative classification based on activity
            if label is None:
                if any(
                    word in comment
                    for word in ["substrate", "efflux", "transported", "p-gp sub"]
                ):
                    label = 1
                elif any(
                    word in comment
                    for word in [
                        "nonsubstrate",
                        "non-efflux",
                        "not transported",
                        "inhibitor",
                    ]
                ):
                    label = 0
                elif pd.notna(pchembl) and pchembl > 5.0:
                    label = 1
                elif pd.notna(pchembl):
                    label = 0

            if label is not None:
                curated.append(
                    {
                        "smiles": smiles,
                        "label": label,
                        "target_type": row.get("target_type"),
                        "pchembl_value": pchembl,
                        "assay_id": row.get("assay_id"),
                    }
                )

        curated_df = pd.DataFrame(curated)

        if not curated_df.empty:
            # Remove duplicates
            curated_df = curated_df.dropna(subset=["smiles"])
            curated_df = curated_df.drop_duplicates(subset=["smiles"])

            # Save curated data
            output_path = self.output_dir / "curated" / "efflux_curated.csv"
            curated_df.to_csv(output_path, index=False)
            logger.info(
                f"Curated {len(curated_df)} Efflux compounds "
                f"({curated_df['label'].sum()} substrates)"
            )

        return curated_df

    def load_cns_drugs_from_drugbank(self, drugbank_path: str) -> pd.DataFrame:
        """
        Load CNS drug classification from DrugBank.

        Requires DrugBank XML file (license required).

        Args:
            drugbank_path: Path to DrugBank XML file

        Returns:
            DataFrame with CNS drug labels
        """
        logger.info("Loading CNS drugs from DrugBank...")

        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(drugbank_path)
            root = tree.getroot()

            drugs = []

            for drug in root.findall(".//drug"):
                name = drug.find("name")
                smiles = drug.find(".//canonical-smiles")
                categories = drug.findall(".//category")

                if name is None or smiles is None:
                    continue

                # Check if CNS drug
                is_cns = False
                for cat in categories:
                    category = cat.get("category", "").lower()
                    if any(
                        word in category
                        for word in [
                            "central nervous system",
                            "cns",
                            "neurological",
                            "psychiatric",
                        ]
                    ):
                        is_cns = True
                        break

                drugs.append(
                    {"name": name.text, "smiles": smiles.text, "cns_label": int(is_cns)}
                )

            df = pd.DataFrame(drugs)
            df = df.dropna(subset=["smiles"])
            df = df.drop_duplicates(subset=["smiles"])

            # Save curated data
            output_path = self.output_dir / "curated" / "cns_drugs_curated.csv"
            df.to_csv(output_path, index=False)
            logger.info(
                f"Loaded {len(df)} CNS drugs " f"({df['cns_label'].sum()} CNS-active)"
            )

            return df

        except Exception as e:
            logger.error(f"Error loading DrugBank data: {e}")
            return pd.DataFrame()

    def create_synthetic_labels_from_b3db(
        self, b3db_path: str = "data/b3db.csv"
    ) -> pd.DataFrame:
        """
        Create synthetic transport labels from B3DB data.

        This is a fallback option if external transport data is unavailable.
        Uses physicochemical properties to predict likely transport mechanism.

        Rules (simplified from literature):
        - Passive diffusion: Low MW, low TPSA, high LogP
        - Influx likely: Contains specific substructures (amines, carboxylates)
        - Efflux likely: High MW, specific substructures (e.g., MACCS8)
        """
        logger.info("Creating synthetic transport labels from B3DB...")

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, MACCSkeys

            df = pd.read_csv(b3db_path)

            # Assuming B3DB has SMILES and BBB permeability label
            if "smiles" not in df.columns:
                logger.error("B3DB must have 'smiles' column")
                return pd.DataFrame()

            labeled = []

            for _, row in df.iterrows():
                smiles = row.get("smiles")
                if not smiles or pd.isna(smiles):
                    continue

                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue

                    # Calculate properties
                    mw = Descriptors.MolWt(mol)
                    tpsa = Descriptors.TPSA(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)

                    # Get MACCS keys
                    maccs = MACCSkeys.GenMACCSKeys(mol)
                    maccs_array = np.array(maccs)

                    # Predict transport mechanism
                    # Passive: Lipinski's rule compliant, low TPSA
                    if mw < 500 and tpsa < 90 and logp > 0 and logp < 5:
                        mechanism = "passive"
                    # Efflux: Check for MACCS8 (beta-lactam) or high MW
                    elif maccs_array[7] == 1 or mw > 500:
                        mechanism = "efflux"
                    # Influx: Check for MACCS43 (two amines) or MACCS36 (sulfur)
                    elif maccs_array[42] == 1 or maccs_array[35] == 1:
                        mechanism = "influx"
                    # Mixed/uncertain
                    else:
                        mechanism = "mixed"

                    labeled.append(
                        {
                            "smiles": smiles,
                            "bbb_label": row.get("BBB_permeable", row.get("label", 1)),
                            "mechanism": mechanism,
                            "mw": mw,
                            "tpsa": tpsa,
                            "logp": logp,
                            "hbd": hbd,
                            "hba": hba,
                        }
                    )

                except Exception as e:
                    logger.debug(f"Error processing SMILES {smiles}: {e}")
                    continue

            labeled_df = pd.DataFrame(labeled)

            # Save
            output_path = self.output_dir / "curated" / "b3db_synthetic_mechanisms.csv"
            labeled_df.to_csv(output_path, index=False)
            logger.info(f"Created synthetic labels for {len(labeled_df)} compounds")

            return labeled_df

        except Exception as e:
            logger.error(f"Error creating synthetic labels: {e}")
            return pd.DataFrame()

    def collect_all_transport_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect all transport mechanism data.

        Returns:
            Dictionary with curated datasets for each mechanism
        """
        logger.info("Starting comprehensive transport data collection...")

        results = {}

        # 1. Collect PAMPA data
        logger.info("\n" + "=" * 50)
        logger.info("Collecting PAMPA (passive diffusion) data...")
        pampa_raw = self.fetch_chembl_data(
            target_type="PAMPA",
            assay_keywords=["PAMPA", "parallel artificial membrane", "BBB permeability"],
        )
        results["pampa"] = self.curate_pampa_data(pampa_raw)

        # 2. Collect Influx (SLC) data
        logger.info("\n" + "=" * 50)
        logger.info("Collecting Influx (SLC transporter) data...")
        influx_raw = self.fetch_chembl_data(
            target_type="Influx",
            assay_keywords=[
                "SLC22",
                "SLCO",
                "OAT",
                "OCT",
                "OATP",
                "solute carrier",
            ],
        )
        results["influx"] = self.curate_influx_data(influx_raw)

        # 3. Collect Efflux (ABC) data
        logger.info("\n" + "=" * 50)
        logger.info("Collecting Efflux (ABC transporter) data...")
        efflux_raw = self.fetch_chembl_data(
            target_type="Efflux",
            assay_keywords=[
                "ABCB1",
                "MDR1",
                "P-glycoprotein",
                "P-gp",
                "ABCG2",
                "BCRP",
                "ABCC1",
                "MRP1",
                "efflux",
            ],
        )
        results["efflux"] = self.curate_efflux_data(efflux_raw)

        # 4. Load CNS drugs (if DrugBank available)
        logger.info("\n" + "=" * 50)
        logger.info("Loading CNS drug data...")
        drugbank_path = "data/DrugBank.xml"
        if os.path.exists(drugbank_path):
            results["cns"] = self.load_cns_drugs_from_drugbank(drugbank_path)
        else:
            logger.warning("DrugBank file not found, using synthetic labels...")
            results["cns"] = None

        # 5. Create synthetic labels from B3DB as fallback
        logger.info("\n" + "=" * 50)
        logger.info("Creating synthetic mechanism labels from B3DB...")
        results["b3db_synthetic"] = self.create_synthetic_labels_from_b3db()

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("Data Collection Summary:")
        logger.info("=" * 50)
        for name, df in results.items():
            if df is not None and not df.empty:
                logger.info(f"{name}: {len(df)} compounds")
                if "label" in df.columns:
                    pos_count = df["label"].sum()
                    logger.info(
                        f"  - Positive: {pos_count} ({pos_count/len(df)*100:.1f}%)"
                    )
                    logger.info(
                        f"  - Negative: {len(df)-pos_count} ({(len(df)-pos_count)/len(df)*100:.1f}%)"
                    )

        return results


def main():
    """Main function to test data collection."""
    collector = TransportDataCollector()

    # Collect all data
    results = collector.collect_all_transport_data()

    print("\nData collection complete!")
    print(f"Results saved to: {collector.output_dir}")


if __name__ == "__main__":
    main()
