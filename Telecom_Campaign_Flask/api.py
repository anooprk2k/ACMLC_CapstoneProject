from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn import preprocessing
import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from xgboost import XGBClassifier


from sqlalchemy import null
warnings.filterwarnings("ignore")
# import xgboost


app = Flask(__name__)

with open('Model/age_mod1_lr.pkl','rb') as fp:
	age_model = pickle.load(fp)

with open('Model/gender_logistic.pkl','rb') as fp:
	gender_model = pickle.load(fp)

with open('Model/scaler.pkl','rb') as fp:
	scaler = pickle.load(fp)

@app.route('/')
def home():
	return render_template('StaticPrediction.html')

@app.route('/predict',methods=['POST'])
def predict():
	df = pd.read_parquet(f'Data/scenario1_test')
	test_df = df.sample(50)

	identifier_cols = ['device_id','gender','age','age_grp','train_test_flag','gender_flag','age_grp_flag']
	feature_cols = list(set(test_df.columns)-set(identifier_cols))
	feature_cols = list(set(feature_cols)-{'event_count','avg_active_apps'} | {'event_count_scaled','avg_active_apps_scaled'})

	predict_type = request.form.get('Type')

	test_df[['event_count_scaled','avg_active_apps_scaled']] = scaler.transform(test_df[['event_count','avg_active_apps']])
	Input = test_df[feature_cols]

	def deflag_age_grp(x):
		if x==0:
			return '0-24'
		elif x==1:
			return '25-32'
		else:
			return '32+'

	def choose_age_campaign(x):
		if x=='0-24':
			return "Campaign 4"
		elif x=='25-32':
			return "Campaign 5"
		else:
			return "Campaign 6"
	
	def choose_gender_campaign(x):
		if x=='Female':
			return "Campaign 1"
		elif x=='Male':
			return "Campaign 3"
		else:
			return "Unable to predict"

	if predict_type == '1':
		test_df['predict_flag'] = age_model.predict(Input)
		test_df['age_grp_predicted'] = test_df['predict_flag'].apply(deflag_age_grp)
		test_df['campaign_type'] = test_df['age_grp_predicted'].apply(choose_age_campaign)
		table = test_df[['device_id', 'age_grp_predicted', 'campaign_type']].to_html(index=False)
		return render_template('StaticPrediction.html', table=table)
	elif predict_type == '2':
		test_df['predict_class1_prob'] = gender_model.predict_proba(Input)[:,1]

		def gender_ks(x):
			ks = pd.read_csv('Model/gender_mod1_lr_KS.csv')
			female_maxprob = ks.loc[2,'max_prob'] #female as class 0
			male_minprob = ks.loc[7,'min_prob'] #male as class 1
			if(x<female_maxprob):
				return 'Female'
			elif(x>male_minprob):
				return 'Male'
			else:
				return 'Unable to predict'


		test_df['gender_predicted'] = test_df['predict_class1_prob'].apply(gender_ks)
		test_df['campaign_type'] = test_df['gender_predicted'].apply(choose_gender_campaign)
		table = test_df[['device_id', 'gender_predicted', 'campaign_type']].to_html(index=False)
		return render_template('StaticPrediction.html', table=table)
	else:
		prediction = null

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
