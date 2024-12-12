import json
import torch
import io
import torchvision
import pandas as pd
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from PIL import Image
from torchvision import transforms
import pandas as pd
from transformers import AutoProcessor, VisionEncoderDecoderModel
import requests
import json
from PIL import Image
import torch
import argparse
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
# ROOT = os.getcwd()
ROOT = "/Data/FSL_codebase/FSL_UB_API"
artifact = 'artifact'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORY_MAPPING_PATH = os.path.join(ROOT, artifact, "notes.json")
MODEL_PATH = os.path.join(ROOT, artifact, "ub__100.pth")
UB_FORM_KEY_MAPPING = os.path.join(ROOT, artifact, "FSL_Forms_Keys.xlsx")
UB_AVERAGE_COORDINATE_PATH = os.path.join(ROOT, artifact, "average_coordinates_ub.xlsx")
KEYS_FROM_OLD = ['38_InsAddr1',
  '38_InsCity',
  '38_InsPostCode',
  '38_InsState',]
BBOX_DONUT_Mapping_Dict = {'1_Bill_Prov_Details': ['1_BillProvAddr1',
  '1_BillProvCity',
  '1_BillProvOrgName',
  '1_BillProvPhoneNumber',
  '1_BillProvPostCode',
  '1_BillProvState',
  '1_MediBillProvAddr1',
  '1_MediBillProvAddr2',
  '1_MediBillProvCity',
  '1_MediBillProvCredential',
  '1_MediBillProvFullName',
  '1_MediBillProvFullPost',
  '1_MediBillProvLName',
  '1_MediBillProvMI',
  '1_MediBillProvOrgName',
  '1_MediBillProvPhoneNumber',
  '1_MediBillProvPostCode',
  '1_MediBillProvPostCodeExt',
  '1_MediBillProvPrefix',
  '1_MediBillProvState',
  '1_MediBillProvSuffix',
  '1_MediBillProvvFName'],
 '2_PO_BOX': ['2_PayToProvAddr1',
  '2_PayToProvAddr2',
  '2_PayToProvCity',
  '2_PayToProvFedIdCode',
  '2_PayToProvFullPost',
  '2_PayToProvNPI',
  '2_PayToProvOrgName',
  '2_PayToProvPhoneNumber',
  '2_PayToProvPostCode',
  '2_PayToProvPostCodeExt',
  '2_PayToProvState'],
 '3_PAT CNTL AND MED REC': ['3_PatControlNumber', '3_PatMedicalRecordNo', 'MedicaidAmount'],
 '4_TYPE OF BILL': '4_TypeofBill',
 '5_FED_TAX_NO': ['5_BillProvFedIdCode', '5_MedicaidTaxId'],
 '6_STATEMENT_COVERS_PERIOD': ['6_StateCovFrom', '6_StateCovTo'],
 '7_Label': '',
 '8_PATIENT_NAME': ['8_PatFullName','8_PatIdentifier'],
 '9_PATIENT_ADDRESS': ['9_PatAddr1',
  '9_PatCity',
  '9_PatPostCode',
  '9_PatState'],
 '10_PATIENT_DOB': '10_PatDOB',
 '11_PATIENT_SEX': '11_PatSex',
 '12_ADMISSION_DETAILS': ['12_PatAdmissionDate',
  '13_PatAdmissionHour',
  '4_AdmissionType',
  '15_PatAdmissionSource',
  '14_PatAdmissionType'],
 '16_DISCHARGE_DETAILS': ['16_PatDischargeHour', '17_PatDischargeStatus'],
 '18_28_CONDITION_CODES': ['18_ConditionCode',
  '19_ConditionCode',
  '20_ConditionCode',
  '21_ConditionCode',
  '22_ConditionCode',
  '23_ConditionCode',
  '24_ConditionCode',
  '25_ConditionCode',
  '26_ConditionCode',
  '27_ConditionCode',
  '28_ConditionCode'],
 '29_State': '29_AccdtSt',
 '30_Label': '',
 '31_OCCURENCE_CODE_DATE': ['31_OccCode', '31_OccDate'],
 '32_OCCURENCE_CODE_DATE': ['32_OccCode', '32_OccDate'],
 '33_OCCURENCE_CODE_DATE': ['33_OccCode', '33_OccDate'],
 '34_OCCURENCE_CODE_DATE': ['34_OccCode', '34_OccDate'],
 '35_OCCURENCE_CODE_SPAN': ['35_OccFromDate',
  '35_OccSpanCode',
  '35_OccThruDate',
  'OccThruDate', 
  'OccSpanCode', 'OccFromDate'],
 '36_OCCURENCE_CODE_SPAN': ['36_OccFromDate',
  '36_OccSpanCode',
  '36_OccThruDate'],
 '37_Label': ' ',
 '38_INSURRENCE_DETAILS': ['38_InsAddr1',
  '38_InsCity',
  '38_InsFName',
  '38_InsFullName',
  '38_InsFullPost',
  '38_InsLName',
  '38_InsMI',
  '38_InsOrgName',
  '38_InsPostCode',
  '38_InsPrefix',
  '38_InsState',
  '38_InsSuffix'],
 '39_VALUE_CODES': ['39_ValueAmount', '39_ValueCode'],
 '40_VALUE_CODES': ['40_ValueAmount', '40_ValueCode'],
 '41_VALUE_CODES': ['41_ValueAmount', '41_ValueCode'],
 '42_49_TABLE': ['42_MedicaidPaidAmount',
  '42_RevCD',
  '43_Description',
  '43_NDC',
  '43_NDCUnits',
  '43_NDCUnitsQual',
  '44_HCPCS',
  '44_Modifier',
  '45_ServDate',
  '46_Unit',
  '47_Charges',
  '48_NonCovChrg'],
 '50_PAYER_NAME': ['50_Payer', '50_PayerIndicator'],
 '51_Health_Plan_No': '51_ProvId',
 '52_Label': '52_RelInfo',
 '53_ASG_BEN': '53_AsgBen',
 '54_PRIOR_PAYMENTS': '54_PriorPay',
 '55_Est_Amount_Due': '55_EstAmount',
 '56_NPI': ['56_BillProvNPI', '56_MediBillProvNPI'],
 '57_OTHER_PRV_ID': ['57_BillProvOtherIdQual', '57_BillProvOtherId'],
 '58_INSURED_NAME': '58_InsFullName',
 '59_P.REL': '59_InsPatRel',
 '60_INSURED_UNIQUE_ID': '60_InsIdCode',
 '61_GROUP_NAME': '61_GrpName',
 '62_INSURANCE_GROUP_NO': '62_GrpNum',
 '63_TREATMENT_AUTHORIZATION_CODES': '63_TreatAutoCode',
 '64_Document_Control_Number': '64_DocCtrlNum',
 '65_EMPLOYER_NAME': '65_EmpName',
 '67_DX': ['67_PrinDiagCode',
  '67_PrinPOA',
  '67A_OtherDiagCode',
  '67A_OtherDiagPOA',
  '67B_OtherDiagCode',
  '67B_OtherDiagPOA',
  '67C_OtherDiagCode',
  '67C_OtherDiagPOA',
  '67D_OtherDiagCode',
  '67D_OtherDiagPOA',
  '67E_OtherDiagCode',
  '67E_OtherDiagPOA',
  '67F_OtherDiagCode',
  '67F_OtherDiagPOA',
  '67G_OtherDiagCode',
  '67G_OtherDiagPOA',
  '67H_OtherDiagCode',
  '67H_OtherDiagPOA',
  '67I_OtherDiagCode',
  '67I_OtherDiagPOA',
  '67J_OtherDiagCode',
  '67J_OtherDiagPOA',
  '67K_OtherDiagCode',
  '67K_OtherDiagPOA',
  '67L_OtherDiagCode',
  '67L_OtherDiagPOA',
  '67M_OtherDiagCode',
  '67M_OtherDiagPOA',
  '67N_OtherDiagCode',
  '67N_OtherDiagPOA',
  '67O_OtherDiagCode',
  '67O_OtherDiagPOA',
  '67P_OtherDiagCode',
  '67P_OtherDiagPOA',
  '67Q_OtherDiagCode',
  '67Q_OtherDiagPOA',
  '66_ICDCode'],
 '68_Label': ['68A_Other', '68B_Other'],
 '69_ADMT_DX': '69_AdmDiagCode',
 '70_PATIENT_REASON_DX': ['70A_PatReasonCode',
  '70B_PatReasonCode',
  '70C_PatReasonCode'],
 '71_ PPS_CODE': '71_PPSCode',
 '72_ECI': ['72A_Ecode', '72B_Ecode', '72C_Ecode'],
 '73_Label': '',
 '74 PRINCIPAL_OTHER_PROCEDURE_CODE': ['74_PrincipleProcCode',
  '74_PrincipleProcDate',
  '74A_OtherProcCode',
  '74A_OtherProcDate',
  '74B_OtherProcCode',
  '74B_OtherProcDate',
  '74C_OtherProcCode',
  '74C_OtherProcDate',
  '74D_OtherProcCode',
  '74D_OtherProcDate',
  '74E_OtherProcCode',
  '74E_OtherProcDate'],
 '75_Label': '',
 '76_ATTENDING_DETAILS': ['76_AttProvFName',
  '76_AttProvLName',
  '76_AttProvNPI',
  '76_AttProvOrgName',
  '76_AttProvOthId',
  '76_AttProvQualId', '76_AttProvFNameDE',
 '76_AttProvFullName',
 '76_AttProvLNameDE'],
 '77_OPERATING_DETAILS': ['77_OpProvFName',
  '77_OpProvLName',
  '77_OpProvNPI',
  '77_OpProvOrgName',
  '77_OpProvOthId',
  '77_OpProvQualId', '77_OpProvFNameDE',
 '77_OpProvFullName',
 '77_OpProvLNameDE'],
 '78_OTHER_DETAILS': ['78_OthProvFName',
  '78_OthProvLName',
  '78_OthProvNPI',
  '78_OthProvNPIQual',
  '78_OthProvOrgName',
  '78_OthProvOthId',
  '78_OthProvQualId', '78_OthProvFNameDE',
 '78_OthProvFullName',
 '78_OthProvLNameDE',
 '78_OthProvNameQual'],
 '79_OTHER_DETAILS': ['79_OthProvFName',
  '79_OthProvLName',
  '79_OthProvNPI',
  '79_OthProvNPIQual',
  '79_OthProvOrgName',
  '79_OthProvOthId',
  '79_OthProvQualId', '79_OthProvFNameDE',
 '79_OthProvFullName',
 '79_OthProvLNameDE',
 '79_OthProvNameQual'],
 '80_REMARKS': ['80_NYSurCharge','MedicaidFlag'],
 '81_Label': ['81A_AddtlCodeValue','81A_AddtlCodeInd',
  '81B_AddtlCodeInd',
  '81B_AddtlCodeValue',
  '81C_AddtlCodeInd',
  '81C_AddtlCodeValue',
  '81D_AddtlCodeInd',
  '81D_AddtlCodeValue']}
average_coordinates_ub_df = pd.read_excel(UB_AVERAGE_COORDINATE_PATH)
key_mapping = pd.read_excel(UB_FORM_KEY_MAPPING)
mapping_dict = key_mapping.set_index('Key_Name').to_dict()['Modified_key']
reverse_mapping_dict = {v: k for k, v in mapping_dict.items()}


class DentalRoiPredictor:
    def __init__(self, model_path, category_mapping_path=CATEGORY_MAPPING_PATH):
        self.category_mapping = self._load_category_mapping(category_mapping_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = len(self.category_mapping) + 1
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def _load_category_mapping(self, category_mapping_path):
        with open(category_mapping_path) as f:
            return {c['id'] + 1: c['name'] for c in json.load(f)['categories']}

    def _get_transforms(self):
        return T.Compose([T.ToDtype(torch.float, scale=True), T.ToPureTensor()])

    def _apply_nms(self, orig_prediction, iou_thresh=0.3):
        keep = torchvision.ops.nms(
            orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
        final_prediction = orig_prediction
        final_prediction['boxes'] = final_prediction['boxes'][keep]
        final_prediction['scores'] = final_prediction['scores'][keep]
        final_prediction['labels'] = final_prediction['labels'][keep]
        return final_prediction

    def _postprocessing_annotation(self, df):
        x1_8_patient = df.loc[df['label'] == '8_PATIENT_NAME', 'x1'].mean()
        x1_7_label = df.loc[df['label'] == '7_Label', 'x1'].mean()

        x0_31_occurence_code_date = df.loc[df['label'] == '31_OCCURENCE_CODE_DATE', 'x0'].mean()
        x0_39_value_codes = df.loc[df['label'] == '39_VALUE_CODES', 'x0'].mean()

        x0_63_treatment_auth_codes = df.loc[df['label'] == '63_TREATMENT_AUTHORIZATION_CODES', 'x0'].mean()
        x0_68_label = df.loc[df['label'] == '68_Label', 'x0'].mean()

        x1_80_remarks = df.loc[df['label'] == '80_REMARKS', 'x1'].mean()
        x0_78_other_details = df.loc[df['label'] == '78_OTHER_DETAILS', 'x0'].mean()

        x1_69_amdt_dx = df.loc[df['label'] == '69_ADMT_DX', 'x1'].mean()
        x0_71_pps_code = df.loc[df['label'] == '71_ PPS_CODE', 'x0'].mean()

        x1_69_amdt_dx = df.loc[df['label'] == '69_ADMT_DX', 'x1'].mean()
        x0_71_pps_code = df.loc[df['label'] == '71_ PPS_CODE', 'x0'].mean()

        x0_69_amdt_dx = df.loc[df['label'] == '69_ADMT_DX', 'x0'].mean()
        x0_75_label = df.loc[df['label'] == '75_Label', 'x0'].mean()

        x0_69_amdt_dx = df.loc[df['label'] == '69_ADMT_DX', 'x0'].mean()
        x0_75_label = df.loc[df['label'] == '75_Label', 'x0'].mean()

        # 51_Health_Plan_No
        x0_51_health_plan_no = df.loc[df['label'] == '51_Health_Plan_No', 'x0'].mean()

        x0_5_fed_tax_no = df.loc[df['label'] == '5_FED_TAX_NO', 'x0'].mean()
        x0_4_type_bill = df.loc[df['label'] == '4_TYPE OF BILL', 'x0'].mean()

        # # Apply post-processing for '9_PATIENT_ADDRESS' class
        df.loc[(df['label'] == '9_PATIENT_ADDRESS'), 'x0'] = x1_8_patient
        df.loc[(df['label'] == '9_PATIENT_ADDRESS'), 'x1'] = x1_7_label


        # # Apply post-processing for '38_INSURRENCE_DETAILS' class
        df.loc[(df['label'] == '38_INSURRENCE_DETAILS'), 'x0'] = x0_31_occurence_code_date
        df.loc[(df['label'] == '38_INSURRENCE_DETAILS'), 'x1'] = x0_39_value_codes

        # # Apply post-processing for '67_DX' class
        df.loc[(df['label'] == '67_DX'), 'x0'] = x0_63_treatment_auth_codes
        df.loc[(df['label'] == '67_DX'), 'x1'] = x0_68_label

        # # Apply post-processing for '81_Label' class
        df.loc[(df['label'] == '81_Label'), 'x0'] = x1_80_remarks
        df.loc[(df['label'] == '81_Label'), 'x1'] = x0_78_other_details

        # # 42_49_TABLE
        df.loc[(df['label'] == '42_49_TABLE'), 'x0'] = x0_31_occurence_code_date
        df.loc[(df['label'] == '42_49_TABLE'), 'x1'] = x1_7_label

        ## 70_PATIENT_REASON_DX
        df.loc[(df['label'] == '70_PATIENT_REASON_DX'), 'x0'] = x1_69_amdt_dx
        df.loc[(df['label'] == '70_PATIENT_REASON_DX'), 'x1'] = x0_71_pps_code

        ## 74 PRINCIPAL_OTHER_PROCEDURE_CODE
        df.loc[(df['label'] == '74 PRINCIPAL_OTHER_PROCEDURE_CODE'), 'x0'] = x0_69_amdt_dx
        df.loc[(df['label'] == '74 PRINCIPAL_OTHER_PROCEDURE_CODE'), 'x1'] = x0_75_label

        ## 50_PAYER_NAME
        df.loc[(df['label'] == '50_PAYER_NAME'), 'x0'] = x0_31_occurence_code_date
        df.loc[(df['label'] == '50_PAYER_NAME'), 'x1'] = x0_51_health_plan_no

        ## 3_PAT CNTL AND MED REC
        df.loc[(df['label'] == '3_PAT CNTL AND MED REC'), 'x0'] = x0_5_fed_tax_no
        df.loc[(df['label'] == '3_PAT CNTL AND MED REC'), 'x1'] = x0_4_type_bill

        return df

    def predict_image(self, image):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(device)
        # pil_image = Image.open(image_path)
        # to_tensor = transforms.ToTensor()
        # image = to_tensor(pil_image)
        image_tensor = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = self.model(image_tensor)
        return predictions

    def predict_and_get_dataframe(self, image_path, image,  iou_thresh=0.5):
        predictions = self.predict_image(image)
        pred = predictions[0]
        pred_nms = self._apply_nms(pred, iou_thresh=iou_thresh)

        pred_dict = {
            'boxes': pred_nms['boxes'].cpu().numpy(),
            'labels': pred_nms['labels'].cpu().numpy(),
            'scores': pred_nms['scores'].cpu().numpy()
        }

        boxes_flat = pred_dict['boxes'].reshape(-1, 4)
        labels_flat = pred_dict['labels'].reshape(-1)
        scores_flat = pred_dict['scores'].reshape(-1)

        class_names = [self.category_mapping[label_id] for label_id in labels_flat]
        num_predictions = len(boxes_flat)
        file_name = [image_path.split(".")[0]] * num_predictions

        infer_df = pd.DataFrame({
            'file_name': file_name,
            'x0': boxes_flat[:, 0],
            'y0': boxes_flat[:, 1],
            'x1': boxes_flat[:, 2],
            'y1': boxes_flat[:, 3],
            'label': labels_flat,
            'class_name': class_names,
            'score': scores_flat
        })

        post_processed_df = self._postprocessing_annotation(infer_df)
        return post_processed_df

# Load the RPI model
frcnn_predictor = DentalRoiPredictor(MODEL_PATH)


def roi_model_inference(image_path, image):
    result_df = frcnn_predictor.predict_and_get_dataframe(image_path, image)
    max_score_indices = result_df.groupby('class_name')['score'].idxmax()
    result_df = result_df.loc[max_score_indices]
    return result_df

def run_prediction_donut(image, model, processor):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=2,
        epsilon_cutoff=6e-4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = prediction.replace("<one>", "1")
    prediction = processor.token2json(prediction)
    return prediction, outputs

def split_and_expand(row):
    keys = [row['Key']] * len(row['Value'].split(';'))
    values = row['Value'].split(';')
    return pd.DataFrame({'Key': keys, 'Value': values})


def load_model(device):
    try:
        # Load Non table model
        non_table_processor = AutoProcessor.from_pretrained("Laskari-Naveen/UB_Model_I")
        non_table_model = VisionEncoderDecoderModel.from_pretrained("Laskari-Naveen/UB_Model_I", cache_dir= "/Data/FSL_codebase/FSL_UB_API/UB_Model_1")
        non_table_model.eval().to(device)
        print("Non Table Model loaded successfully")
        table_processor = AutoProcessor.from_pretrained("Laskari-Naveen/UB_Table_Model")
        table_model = VisionEncoderDecoderModel.from_pretrained("Laskari-Naveen/UB_Table_Model", cache_dir= "/Data/FSL_codebase/FSL_UB_API/UB_Table_Model")
        table_model.eval().to(device)
        print("Table Model loaded successfully")
        old_non_table_processor = AutoProcessor.from_pretrained("Laskari-Naveen/UB_2_P1")
        old_non_table_model = VisionEncoderDecoderModel.from_pretrained("Laskari-Naveen/UB_2_P1", cache_dir= "/Data/FSL_codebase/FSL_UB_API/UB_Table_Model")
        old_non_table_model.eval().to(device)
    except Exception as e:
        print(f"Model Loading failed !!! with error {e}")
    return non_table_processor, non_table_model, table_processor, table_model, old_non_table_processor, old_non_table_model


def convert_predictions_to_df(prediction):
    expanded_df = pd.DataFrame()
    result_df_each_image = pd.DataFrame()    
    each_image_output = pd.DataFrame(list(prediction.items()), columns=["Key", "Value"])
    #print(each_image_output.head(1))
    try:    
        expanded_df = pd.DataFrame(columns=['Key', 'Value'])
        for index, row in each_image_output[each_image_output['Value'].str.contains(';')].iterrows():
            expanded_df = pd.concat([expanded_df, pd.DataFrame(split_and_expand(row))], ignore_index=True)

        result_df_each_image = pd.concat([each_image_output, expanded_df], ignore_index=True)
        result_df_each_image = result_df_each_image.drop(result_df_each_image[result_df_each_image['Value'].str.contains(';')].index)

        for old_key, new_key in reverse_mapping_dict.items():
            result_df_each_image["Key"].replace(old_key, new_key, inplace=True)
    except Exception as e:
        print(f"Error in convert_predictions_to_df: {e}")
        pass
        
    return result_df_each_image

# def plot_bounding_boxes(image, df, enable_title = False):
#     image = image.permute(1,2,0)
#     colors = ['red', 'blue', 'green', 'orange', 'purple', 'magenta', 'brown']
#     fig, ax = plt.subplots(1, figsize=(50, 50))
#     ax.set_aspect('auto')
#     ax.imshow(image)
#     for index, row in df.iterrows():
#         class_name = row['class_name']
#         x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
#         box_color = random.choice(colors)
#         rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.5, edgecolor=box_color, facecolor='none')
#         ax.add_patch(rect)

#         if enable_title:
#             ax.text(x0, y0, class_name, color=box_color, fontsize=9, weight='bold')
#     ax.axis('off')
#     plt.show()

def map_result1(dict1, dict2):
    result_dict_1 = {}
    for key, value in dict1.items():
        if key in dict2:
            mapping_keys = dict2[key] if isinstance(dict2[key], list) else [dict2[key]]
            for mapping_key in mapping_keys:
                result_dict_1[mapping_key] = value
    return result_dict_1

def map_result2(dict1, dict2):
    result_dict_2 = {}
    for key, value in dict1.items():
        if key in dict2:
            mapping_keys = dict2[key] if isinstance(dict2[key], list) else [dict2[key]]
            for mapping_key in mapping_keys:
                result_dict_2[key] = {
                    "Mapping_key": mapping_keys,
                    "coordinates": value
                }
    return result_dict_2

def map_result1_final_output(result_dict_1, additional_info_dict):
    updated_result_dict_1 = {}

    # Iterate over additional_info_dict
    for key, additional_info in additional_info_dict.items():
        # Check if the key exists in result_dict_1
        if key in result_dict_1:
            coordinates = result_dict_1[key]
        else:
            # If the key is missing in result_dict_1, set coordinates to None
            coordinates = None

        # Store the coordinates and additional_info in updated_result_dict_1
        updated_result_dict_1[key] = {"coordinates": coordinates, "text": additional_info}

    return updated_result_dict_1

# def run_application(input_image_folder, output_ROI_folder, output_extraction_folder):
#     root = os.getcwd()
#     os.makedirs(output_ROI_folder, exist_ok=True)
#     os.makedirs(output_extraction_folder, exist_ok=True)
#     # image_list = os.listdir(os.path.join(root, input_image_folder))
#     image_list = os.listdir(input_image_folder)
#     for each_image in tqdm(image_list):
#         image_path = os.path.join(input_image_folder, each_image)
#         pil_image = Image.open(image_path).convert('RGB')
#         to_tensor = transforms.ToTensor()
#         image = to_tensor(pil_image)
        
#         print("Staring ROI extraction")
#         # print(uploaded_file.)
#         fasterrcnn_result_df = roi_model_inference(image_path, image)
#         print("Staring data extraction")
#         prediction, output = run_prediction_donut(image, model, processor)
#         extraction_df = convert_predictions_to_df(prediction)

#         output_ROI_path = os.path.join(root, output_ROI_folder,each_image.split(".")[0]+".xlsx" )
#         fasterrcnn_result_df.to_excel(output_ROI_path, index=False)

#         output_extraction_path = os.path.join(root, output_extraction_folder, each_image.split(".")[0]+".xlsx" )
#         extraction_df.to_excel(output_extraction_path, index=False)


# Load the models
non_table_processor, non_table_model, table_processor, table_model, old_non_table_processor, old_non_table_model = load_model(device)

def merge_donut_output(donut_out_old, donut_out_new, keys_from_old):
    try:
        print("In process of merging from OLD keys")
        old_values_for_keys = donut_out_old.set_index("Key").loc[keys_from_old, 'Value'].to_dict()

        donut_out_new['Value'] = donut_out_new.apply(
            lambda row: old_values_for_keys.get(row['Key'], row['Value']),
            axis = 1
        )

        return donut_out_new[['Key', 'Value']]
    
    except Exception as e:
        raise e



def run_ub_pipeline(image_path: str):
    try:
        # image_path = os.path.join(input_image_folder, each_image)
        pil_image = Image.open(image_path).convert('RGB')
        # pil_image = Image.open(io.BytesIO(image_path)).convert('RGB')
        to_tensor = transforms.ToTensor()
        image = to_tensor(pil_image)

        prediction_non_table, output_non_table = run_prediction_donut(pil_image, non_table_model, non_table_processor)
        prediction_table, output_table = run_prediction_donut(pil_image, table_model, table_processor)
        prediction_non_table.update(prediction_table)
        #print(f'{image_path} {len(prediction_non_table)}')
        donut_out = convert_predictions_to_df(prediction_non_table)

        # print(donut_out)

        # What is this? Is it Mapping the donut keys to XML values? Can't understand.
        # for old_key, new_key in reverse_mapping_dict.items():
        #     donut_out["Key"].replace(old_key, new_key, inplace=True)

        ####### OLD MODEL OUTPUT ########
        prediction_old_non_table, output_old_non_table = run_prediction_donut(pil_image, old_non_table_model, old_non_table_processor)
        donut_out_old = convert_predictions_to_df(prediction_old_non_table)
        ###### MERGE OUTPUT OF OLD AND NEW #####
        donut_out = merge_donut_output(donut_out_old, donut_out, KEYS_FROM_OLD)

        print(donut_out[donut_out['Key'].isin(KEYS_FROM_OLD)])
        print(donut_out['Key'].nunique())

        
        # This is just converting the dataframe to dictionary
        json_data = donut_out.to_json(orient='records')
        data_list = json.loads(json_data)
        output_dict_donut = {}

        # Iterate through the data_list
        for item in data_list:
            key = item['Key']
            value = item['Value'].strip()

            # Check if the key already exists in the output dictionary
            if key in output_dict_donut:
                # If the key exists, append the value to the list of dictionaries
                output_dict_donut[key].append({'value': value})
            else:
                # If the key doesn't exist, create a new list with the current value
                output_dict_donut[key] = [{'value': value}]        

        # This is just doing the ROI inference and converting DF to dict
        res = roi_model_inference(image_path, image)
        df_dict = res.to_dict(orient='records')
        print("OD prediction --->>>", df_dict)
        
        # Implementing the average part here
        # Convert the average coordinates DataFrame to a dictionary for easy access
        average_coordinates_dict = average_coordinates_ub_df.set_index('label').to_dict(orient='index')

        # Get all unique class names
        all_class_names = set(average_coordinates_ub_df['label'])

        # Initialize the output dictionary
        output_dict_det = {}

        # Iterate over all class names
        for class_name in all_class_names:
            # Check if the class name exists in df_dict
            item = next((item for item in df_dict if item['class_name'] == class_name), None)
            if item:
                # If the class name exists, use the coordinates from df_dict
                x1, y1, x2, y2 = item['x0'], item['y0'], item['x1'], item['y1']
            else:
                # If the class name doesn't exist, replace coordinates with average coordinates
                avg_coords = average_coordinates_dict.get(class_name, None)
                if avg_coords:
                    x1 = avg_coords['xmin']
                    y1 = avg_coords['ymin']
                    x2 = avg_coords['xmax']
                    y2 = avg_coords['ymax']
                else:
                    # If average coordinates are not available, set coordinates to NaN
                    x1, y1, x2, y2 = float('nan'), float('nan'), float('nan'), float('nan')

            # Store the coordinates in the output dictionary
            output_dict_det[class_name] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

        # Map the ROI keys with the Donut keys
        result_dict_1 = map_result1(output_dict_det, BBOX_DONUT_Mapping_Dict)
        # result_dict_2 = map_result2(output_dict_det, BBOX_DONUT_Mapping_Dict)
        final_mapping_dict  = map_result1_final_output(result_dict_1, output_dict_donut)
        print(len(final_mapping_dict))
        return {"result": final_mapping_dict}, None
    except Exception as e:
        return None, str(e)

# with open('notes.json') as f:
#     category_mapping = {c['id'] + 1: c['name'] for c in json.load(f)['categories']}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your application description")
    parser.add_argument("input_image_folder", help="Path to the input image folder")
    parser.add_argument("output_ROI_folder", help="Path to the output ROI folder")
    parser.add_argument("output_extraction_folder", help="Path to the output extraction folder")
    args = parser.parse_args()
    processor, model = load_model(device)
    # run_application(args.input_image_folder, args.output_ROI_folder, args.output_extraction_folder)
