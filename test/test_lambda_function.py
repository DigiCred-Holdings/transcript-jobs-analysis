import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import lambda_function
from unittest.mock import patch, MagicMock

def test_lambda_handler():
    event = {}
    context = {
        "summary": "You excel greatly in business, administration, and law, most notably your accounting courses taught you to analyze and interpret financial information effectively. Your strength in applying accounting systems and software, such as financial reporting and accounting information systems, stands out. The 'Principles of Accounting I' course specifically enhanced your ability to understand the foundational concepts of financial reporting and business terminology",
        "student_skill_list": ['national generally accepted accounting principles', 'accounting entries', 'financial analysis', 'accounting techniques', 'financial statements', 'comprehend financial business terminology', 'comply with regulations', 'accounting', 'obtain financial information', 'accounting', 'financial analysis', 'present reports', 'consider economic criteria in decision making', 'make investment decisions', 'conduct performance measurement', 'financial statements', 'financial management', 'cost management', 'manage budgets', 'provide cost benefit analysis reports', 'business knowledge', 'interpret financial statements', 'make strategic business decisions', 'evaluate budgets', 'financial reporting software', 'think critically', 'interpret current data', 'financial analysis', 'financial analysis software', 'financial statements', 'think analytically', 'analyse internal factors of companies', 'present reports', 'analyse external factors of companies', 'inspect tax returns', 'advise on tax planning', 'think analytically', 'inspect taxation documents', 'income tax return preparation software', 'keep up-to-date with regulations', 'tax legislation', 'apply strategic thinking', 'make decisions', 'national generally accepted accounting principles', 'explain accounting records', 'accounting entries', 'financial statements', 'accounting techniques', 'collect financial data', 'ensure compliance with accounting conventions', 'perform asset recognition', 'record corporate property', 'accounting department processes', 'accounting', 'perform asset depreciation', 'check accounting records', 'calculate production costs', 'perform data analysis', 'conduct performance measurement', 'job costing software', 'cost management', 'manage budgets', 'estimate profitability', 'perform cost accounting activities', 'make decisions', 'evaluate budgets', 'explain accounting records', 'accounting entries', 'prepare financial statements', 'handle cash flow', 'ensure compliance with disclosure criteria of accounting information', 'accounting techniques', 'calculate debt costs', 'perform asset recognition', 'ensure compliance with accounting conventions', 'debt classification', 'manage revenue', 'accounting', 'automated information system software', 'financial reporting software', 'financial accounting software', 'ict system integration', 'management information systems mis', 'control systems', 'accounting software', 'collect financial data', 'design control systems', 'auditing software', 'comply with regulations', 'data security principles', 'use accounting systems', 'manage personal professional development', 'personal development', 'develop personal skills', 'business communication', 'career decision-making programs', 'build networks', 'listen actively'],
        "student_skill_groups": {'business, administration and law': 26, 'information skills': 16, 'social and communication skills and competences': 5, 'self-management skills and competences': 3, 'core skills and competences': 5, 'management skills': 16, 'communication, collaboration and creativity': 5, 'thinking skills and competences': 3, 'assisting and caring': 2, 'working with computers': 1, 'engineering, manufacturing and construction': 1, 'information and communication technologies (icts)': 2, 'generic programmes and qualifications': 2},
        "course_id_list": ['CWY000000', 'CWY000001', 'CWY000003', 'CWY000002', 'CWY000004', 'CWY000005', 'CWY000006', 'CWY000007', 'CWY000008']
    }

    response = lambda_function.lambda_handler(event, context)

    assert response['status'] == 200
    assert 'body' in response




test_lambda_handler()