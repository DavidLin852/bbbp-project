"""
BBB渗透机制数据收集脚本
收集PAMPA、Influx、Efflux数据集
"""

import os
import json
import requests
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BBBMechanismDataCollector:
    """BBB渗透机制数据收集器"""

    def __init__(self, output_dir: str = "data/mechanism"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ChEMBL API配置
        self.chembl_api = "https://www.ebi.ac.uk/chembl/api/data"
        self.chembl_mol_api = "https://www.ebi.ac.uk/chembl/api/data/molecule"

        logger.info(f"输出目录: {self.output_dir}")

    def collect_from_chembl_efflux(self, max_results: int = 500) -> pd.DataFrame:
        """
        从ChEMBL收集Efflux (ABC转运体) 数据
        Target: ABCB1 (P-gp), ABCG2 (BCRP), ABCC1-4 (MRPs)
        """
        logger.info("开始收集Efflux数据...")

        # ABC转运体列表
        abc_targets = [
            "CHEMBL240",  # ABCB1 (P-gp)
            "CHEMBL241",  # ABCG2 (BCRP)
            "CHEMBL242",  # ABCC1 (MRP1)
            "CHEMBL243",  # ABCC2 (MRP2)
            "CHEMBL244",  # ABCC3 (MRP3)
            "CHEMBL245",  # ABCC4 (MRP4)
        ]

        all_data = []

        for target_id in abc_targets:
            logger.info(f"查询靶点: {target_id}")

            try:
                # 查询该靶点的所有活性数据
                url = f"{self.chembl_api}/activity?target_chembl_id={target_id}&format=json"

                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                activities = data.get('activities', [])

                for act in activities[:max_results]:
                    try:
                        # 提取数据
                        mol_id = act.get('molecule_chembl_id')
                        if not mol_id:
                            continue

                        # 标准化为efflux ratio
                        # ER >= 5: substrate (efflux+)
                        # ER <= 1: non-substrate (efflux-)
                        # 1 < ER < 5: 中间状态(排除)
                        activity_comment = act.get('activity_comment', '').lower()
                        standard_value = act.get('standard_value')
                        standard_type = act.get('standard_type', '').lower()

                        is_substrate = None
                        if standard_value and 'ratio' in standard_type:
                            er_value = float(standard_value)
                            if er_value >= 5:
                                is_substrate = 1
                            elif er_value <= 1:
                                is_substrate = 0
                            else:
                                continue  # 排除中间值
                        elif 'substrate' in activity_comment:
                            is_substrate = 1
                        elif 'non-substrate' in activity_comment or 'not a substrate' in activity_comment:
                            is_substrate = 0

                        if is_substrate is None:
                            continue

                        all_data.append({
                            'chembl_id': mol_id,
                            'target_id': target_id,
                            'standard_value': standard_value,
                            'standard_type': standard_type,
                            'is_efflux_substrate': is_substrate,
                            'assay_type': act.get('assay_type'),
                            'confidence': act.get('confidence_score', 0)
                        })

                    except Exception as e:
                        logger.debug(f"处理活性数据失败: {e}")
                        continue

                time.sleep(0.1)  # 避免请求过快

            except Exception as e:
                logger.error(f"查询靶点 {target_id} 失败: {e}")
                continue

        df = pd.DataFrame(all_data)

        if not df.empty:
            # 去重: 同一分子有多个结果时,取平均
            df_grouped = df.groupby('chembl_id').agg({
                'is_efflux_substrate': 'mean',
                'target_id': lambda x: ','.join(set(x)),
                'standard_value': lambda x: x.mean(),
                'confidence': 'mean'
            }).reset_index()

            df_grouped['is_efflux_substrate'] = (df_grouped['is_efflux_substrate'] >= 0.5).astype(int)

            output_path = self.output_dir / "efflux_chembl_raw.csv"
            df_grouped.to_csv(output_path, index=False)
            logger.info(f"Efflux数据保存到: {output_path}")
            logger.info(f"总计: {len(df_grouped)} 个分子 (substrate: {df_grouped['is_efflux_substrate'].sum()})")

        return df_grouped if not df.empty else pd.DataFrame()

    def collect_from_chembl_influx(self, max_results: int = 500) -> pd.DataFrame:
        """
        从ChEMBL收集Influx (SLC转运体) 数据
        Target: SLC22, SLCO家族
        """
        logger.info("开始收集Influx数据...")

        # SLC转运体列表
        slc_targets = [
            "SLC22A1",  # OCT1
            "SLC22A2",  # OCT2
            "SLC22A3",  # OCT3
            "SLC22A6",  # OAT1
            "SLC22A8",  # OAT3
            "SLCO1A2",  # OATP1A2
            "SLCO1B1",  # OATP1B1
            "SLCO2B1",  # OATP2B1
        ]

        all_data = []

        for target_name in slc_targets:
            logger.info(f"查询靶点: {target_name}")

            try:
                # 查询该靶点的所有活性数据
                url = f"{self.chembl_api}/activity?target_pref_name={target_name}&format=json"

                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                activities = data.get('activities', [])

                for act in activities[:max_results]:
                    try:
                        mol_id = act.get('molecule_chembl_id')
                        if not mol_id:
                            continue

                        # 判断是否为substrate
                        activity_comment = act.get('activity_comment', '').lower()

                        is_substrate = None
                        if 'substrate' in activity_comment or 'uptake' in activity_comment:
                            is_substrate = 1
                        elif 'non-substrate' in activity_comment or 'not a substrate' in activity_comment:
                            is_substrate = 0
                        elif 'inhibitor' in activity_comment and 'substrate' not in activity_comment:
                            is_substrate = 0

                        if is_substrate is None:
                            continue

                        all_data.append({
                            'chembl_id': mol_id,
                            'target_name': target_name,
                            'is_influx_substrate': is_substrate,
                            'activity_comment': act.get('activity_comment'),
                        })

                    except Exception as e:
                        logger.debug(f"处理活性数据失败: {e}")
                        continue

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"查询靶点 {target_name} 失败: {e}")
                continue

        df = pd.DataFrame(all_data)

        if not df.empty:
            # 去重
            df_grouped = df.groupby('chembl_id').agg({
                'is_influx_substrate': 'mean',
                'target_name': lambda x: ','.join(set(x))
            }).reset_index()

            df_grouped['is_influx_substrate'] = (df_grouped['is_influx_substrate'] >= 0.5).astype(int)

            output_path = self.output_dir / "influx_chembl_raw.csv"
            df_grouped.to_csv(output_path, index=False)
            logger.info(f"Influx数据保存到: {output_path}")
            logger.info(f"总计: {len(df_grouped)} 个分子 (substrate: {df_grouped['is_influx_substrate'].sum()})")

        return df_grouped if not df.empty else pd.DataFrame()

    def collect_from_chembl_pampa(self, max_results: int = 1000) -> pd.DataFrame:
        """
        从ChEMBL收集PAMPA数据
        搜索关键词: PAMPA, parallel artificial membrane
        """
        logger.info("开始收集PAMPA数据...")

        all_data = []

        try:
            # 搜索PAMPA相关的assay
            url = f"{self.chembl_api}/assay?assay_type=P&assay_desc=PAMPA&format=json"

            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            assays = data.get('assays', [])

            logger.info(f"找到 {len(assays)} 个PAMPA assays")

            for assay in assays[:20]:  # 限制assay数量
                assay_id = assay.get('assay_chembl_id')
                logger.info(f"查询assay: {assay_id}")

                try:
                    url = f"{self.chembl_api}/activity?assay_chembl_id={assay_id}&format=json"
                    response = requests.get(url)
                    response.raise_for_status()
                    act_data = response.json()

                    activities = act_data.get('activities', [])

                    for act in activities[:max_results]:
                        try:
                            mol_id = act.get('molecule_chembl_id')
                            if not mol_id:
                                continue

                            # PAMPA渗透性判断
                            # Pe > 4e-6 cm/s: permeable
                            # Pe < 2e-6 cm/s: impermeable
                            standard_value = act.get('standard_value')
                            standard_type = act.get('standard_type', '').lower()

                            if standard_value and ('permeability' in standard_type or 'pe' in standard_type):
                                pe_value = float(standard_value)

                                # 转换为cm/s
                                if pe_value > 4e-6:
                                    is_permeable = 1
                                elif pe_value < 2e-6:
                                    is_permeable = 0
                                else:
                                    continue  # 排除中间值

                                all_data.append({
                                    'chembl_id': mol_id,
                                    'assay_id': assay_id,
                                    'pe_value': pe_value,
                                    'is_pampa_permeable': is_permeable,
                                })

                        except Exception as e:
                            logger.debug(f"处理活性数据失败: {e}")
                            continue

                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"查询assay {assay_id} 失败: {e}")
                    continue

        except Exception as e:
            logger.error(f"PAMPA数据收集失败: {e}")

        df = pd.DataFrame(all_data)

        if not df.empty:
            # 去重: 同一分子取平均
            df_grouped = df.groupby('chembl_id').agg({
                'is_pampa_permeable': 'mean',
                'pe_value': 'mean'
            }).reset_index()

            df_grouped['is_pampa_permeable'] = (df_grouped['is_pampa_permeable'] >= 0.5).astype(int)

            output_path = self.output_dir / "pampa_chembl_raw.csv"
            df_grouped.to_csv(output_path, index=False)
            logger.info(f"PAMPA数据保存到: {output_path}")
            logger.info(f"总计: {len(df_grouped)} 个分子 (permeable: {df_grouped['is_pampa_permeable'].sum()})")

        return df_grouped if not df.empty else pd.DataFrame()

    def fetch_smiles_from_chembl(self, chembl_ids: List[str]) -> Dict[str, str]:
        """
        从ChEMBL获取SMILES
        """
        logger.info(f"获取 {len(chembl_ids)} 个分子的SMILES...")

        smiles_dict = {}
        batch_size = 50

        for i in range(0, len(chembl_ids), batch_size):
            batch = chembl_ids[i:i+batch_size]

            for chembl_id in batch:
                try:
                    url = f"{self.chembl_mol_api}/{chembl_id}.json"
                    response = requests.get(url)
                    response.raise_for_status()
                    data = response.json()

                    mol_structures = data.get('molecule_structures', {})
                    smiles = mol_structures.get('canonical_smiles')

                    if smiles:
                        smiles_dict[chembl_id] = smiles

                except Exception as e:
                    logger.debug(f"获取 {chembl_id} SMILES失败: {e}")
                    continue

                time.sleep(0.05)  # 避免请求过快

        logger.info(f"成功获取 {len(smiles_dict)} 个SMILES")

        return smiles_dict

    def collect_cns_drugs(self) -> pd.DataFrame:
        """
        收集CNS药物数据 (从文献或DrugBank)
        这里使用论文中提到的已知CNS药物列表
        """
        logger.info("开始收集CNS药物数据...")

        # 已知CNS药物列表 (示例)
        cns_drugs = {
            # CNS活性药物
            'Diazepam': 1,
            'Citalopram': 1,
            'Sertraline': 1,
            'Fluoxetine': 1,
            'Paroxetine': 1,
            'Venlafaxine': 1,
            'Amitriptyline': 1,
            'Imipramine': 1,
            'Clomipramine': 1,
            'Bupropion': 1,
            'Donepezil': 1,
            'Memantine': 1,
            'Rivastigmine': 1,
            'Galantamine': 1,
            'Haloperidol': 1,
            'Risperidone': 1,
            'Olanzapine': 1,
            'Quetiapine': 1,
            'Clozapine': 1,
            'Lorazepam': 1,
            'Alprazolam': 1,
            'Zolpidem': 1,
            'Eszopiclone': 1,
            'Modafinil': 1,
            'Methylphenidate': 1,
            'Amphetamine': 1,
            'L-DOPA': 1,
            'Pramipexole': 1,
            'Ropinirole': 1,
            'Carbamazepine': 1,
            'Valproate': 1,
            'Lamotrigine': 1,
            'Levetiracetam': 1,
            'Phenytoin': 1,
            # 非CNS药物
            'Atenolol': 0,
            'Nadolol': 0,
            'Sotalol': 0,
            'Fexofenadine': 0,
            'Loperamide': 0,
            'Ranitidine': 0,
            'Famotidine': 0,
            'Cimetidine': 0,
            'Terfenadine': 0,
            'Astemizole': 0,
        }

        df = pd.DataFrame(list(cns_drugs.items()), columns=['drug_name', 'is_cns_active'])

        output_path = self.output_dir / "cns_drugs_reference.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"CNS药物列表保存到: {output_path}")
        logger.info(f"总计: {len(df)} 个药物 (CNS: {df['is_cns_active'].sum()})")

        return df

    def merge_and_enrich(self, efflux_df: pd.DataFrame, influx_df: pd.DataFrame,
                        pampa_df: pd.DataFrame) -> pd.DataFrame:
        """
        合并所有数据集并添加SMILES
        """
        logger.info("合并数据集...")

        # 收集所有chembl_id
        all_chembl_ids = set()

        if not efflux_df.empty:
            all_chembl_ids.update(efflux_df['chembl_id'].tolist())

        if not influx_df.empty:
            all_chembl_ids.update(influx_df['chembl_id'].tolist())

        if not pampa_df.empty:
            all_chembl_ids.update(pampa_df['chembl_id'].tolist())

        logger.info(f"需要获取SMILES的分子数: {len(all_chembl_ids)}")

        # 批量获取SMILES
        smiles_dict = self.fetch_smiles_from_chembl(list(all_chembl_ids))

        # 创建统一的数据框
        merged_data = []

        for chembl_id in all_chembl_ids:
            row = {
                'chembl_id': chembl_id,
                'smiles': smiles_dict.get(chembl_id, ''),
            }

            if not efflux_df.empty:
                efflux_row = efflux_df[efflux_df['chembl_id'] == chembl_id]
                if not efflux_row.empty:
                    row['is_efflux_substrate'] = efflux_row['is_efflux_substrate'].values[0]

            if not influx_df.empty:
                influx_row = influx_df[influx_df['chembl_id'] == chembl_id]
                if not influx_row.empty:
                    row['is_influx_substrate'] = influx_row['is_influx_substrate'].values[0]

            if not pampa_df.empty:
                pampa_row = pampa_df[pampa_df['chembl_id'] == chembl_id]
                if not pampa_row.empty:
                    row['is_pampa_permeable'] = pampa_row['is_pampa_permeable'].values[0]
                    row['pe_value'] = pampa_row['pe_value'].values[0]

            merged_data.append(row)

        merged_df = pd.DataFrame(merged_data)

        # 过滤掉没有SMILES的分子
        merged_df = merged_df[merged_df['smiles'] != '']

        output_path = self.output_dir / "mechanism_dataset_merged.csv"
        merged_df.to_csv(output_path, index=False)
        logger.info(f"合并数据保存到: {output_path}")
        logger.info(f"总计: {len(merged_df)} 个分子 (有SMILES)")

        return merged_df

    def run_full_collection(self):
        """
        运行完整的数据收集流程
        """
        logger.info("=" * 60)
        logger.info("开始BBB渗透机制数据收集")
        logger.info("=" * 60)

        # 1. 收集Efflux数据
        efflux_df = self.collect_from_chembl_efflux()

        # 2. 收集Influx数据
        influx_df = self.collect_from_chembl_influx()

        # 3. 收集PAMPA数据
        pampa_df = self.collect_from_chembl_pampa()

        # 4. 收集CNS药物参考列表
        cns_df = self.collect_cns_drugs()

        # 5. 合并数据并添加SMILES
        merged_df = self.merge_and_enrich(efflux_df, influx_df, pampa_df)

        logger.info("=" * 60)
        logger.info("数据收集完成!")
        logger.info("=" * 60)

        # 打印统计信息
        logger.info("\n数据集统计:")
        logger.info(f"  Efflux: {len(efflux_df) if not efflux_df.empty else 0} 个分子")
        logger.info(f"  Influx: {len(influx_df) if not influx_df.empty else 0} 个分子")
        logger.info(f"  PAMPA: {len(pampa_df) if not pampa_df.empty else 0} 个分子")
        logger.info(f"  合并: {len(merged_df)} 个分子")

        return efflux_df, influx_df, pampa_df, merged_df


def main():
    """主函数"""
    collector = BBBMechanismDataCollector()
    efflux_df, influx_df, pampa_df, merged_df = collector.run_full_collection()


if __name__ == "__main__":
    main()
