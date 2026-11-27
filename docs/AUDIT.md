# KAIROS two-pass audit

## Results
Independent calculation: 4,050 aligned market rows; 4,013 eligible observations, 28,091 feature values, all target labels and outcome dates. One missing VIX close on 2024-03-02 was forward-filled and two affected feature rows excluded. All 1,203 validation/test predictions and scores reproduced. All 33 notebook cells and three dashboard backend tests passed. Model, labelled dataset and metric/prediction CSV files are byte-identical to the previous package. Run python -m dashboard.audit and python -m pytest -q tests/test_dashboard.py with local exports.

## Weekend sessions
All 19 dates have supporting special-session evidence:
- 2012-01-07: https://www.isesec.com/Admin/Circular/20121170.pdf
- 2012-03-03: https://timesofindia.indiatimes.com/business/india-business/sensex-closes-flat-in-special-trading-session/articleshow/12122540.cms
- 2012-04-28: https://www.cse-india.com/upload/CSE%20Notices%20%26%20Circulars/2012/Notice260412.htm
- 2012-09-08: https://www.business-standard.com/article/markets/nse-to-hold-special-trading-session-on-sep-8-112082700163_1.html
- 2013-05-11: https://indianexpress.com/article/news-archive/web/bse-sensex-closes-40-points-up-in-special-trading/
- 2014-03-22: https://www.cse-india.com/upload/CSE%20Notices%20%26%20Circulars/2014/Notice200314_4.htm
- 2015-02-28: https://www.moneycontrol.com/news/business/markets/budget-2015-nifty-at-1-mnth-closing-high-sensex141-pts-banks-soar-1179505.html
- 2020-02-01: https://indianexpress.com/article/business/market/budget-2020-market-live-bse-sensex-nse-nifty-6245329/
- 2025-02-01: https://upstox.com/news/market-news/stocks/market-wrap-february-1-sensex-nifty-50-end-flat-on-budget-day-consumption-stocks-steal-limelight/article-143751/
- 2026-02-01: https://indianexpress.com/article/business/budget-2026-stock-market-today-live-updates-nifty-sensex-infrastructure-railways-defence-power-msme-share-price-nirmala-sitharaman-speech-10503492/lite/
- 2024-01-20: https://zerodha.com/marketintel/bulletin/367702/live-trading-session-on-saturday-january-20-2024
- 2024-03-02 and 2024-05-18: https://www.reuters.com/world/india/indian-shares-muted-special-saturday-session-2024-05-18/
- 2013-11-03, 2016-10-30, 2019-10-27, 2020-11-14, 2023-11-12, 2026-11-08: https://v2.webnotes.in/muhurat-trading and https://zerodha.com/marketintel/bulletin/461711/muhurat-trading-session-on-account-of-diwali-november-2026/

These support the session dates, not all historical prices or VIX observations. No date was deleted.

## Corrections and limitations
Unified exploratory threshold tolerance, added direct CSV exports, keyed SHAP class widget to date and scope, removed obsolete notebook instructions, forwarded old app entry point. Browser smoke test remains pending because Streamlit is not installed. The new widget key was compiled and inspected, not tested in a real browser. Full exchange archive reconciliation, licensing review and prospective model evaluation remain pending. Test macro F1 0.4091, accuracy 81.59%, balanced accuracy 39.26%, 0/23 major declines detected. No trading reliability established. Scores are uncalibrated and overlapping outcomes are dependent.
