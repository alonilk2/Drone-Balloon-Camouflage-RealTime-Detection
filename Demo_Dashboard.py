import PySimpleGUI as sg
import datetime
import subprocess
import torch
"""
    Dashboard using blocks of information.

    Copyright 2020 PySimpleGUI.org
"""


theme_dict = {'BACKGROUND': '#2B475D',
                'TEXT': '#FFFFFF',
                'INPUT': '#F2EFE8',
                'TEXT_INPUT': '#000000',
                'SCROLL': '#F2EFE8',
                'BUTTON': ('#000000', '#C2D4D8'),
                'PROGRESS': ('#FFFFFF', '#C7D5E0'),
                'BORDER': 1,'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}


now = datetime.datetime.now()
print(torch.cuda.is_available())
# sg.theme_add_new('Dashboard', theme_dict)     # if using 4.20.0.1+
sg.LOOK_AND_FEEL_TABLE['Dashboard'] = theme_dict
sg.theme('Dashboard')

BORDER_COLOR = '#C7D5E0'
DARK_HEADER_COLOR = '#1B2838'
BPAD_TOP = ((20,20), (20, 10))
BPAD_LEFT = ((20,10), (0, 10))
BPAD_LEFT_INSIDE = (0, 10)
BPAD_RIGHT = ((10,20), (10, 20))

top_banner = [[sg.Text('Dashboard'+ ' '*64, font='Any 20', background_color=DARK_HEADER_COLOR),
               sg.Text(now.strftime("%d/%m/%Y    %H:%M"), font='Any 20', background_color=DARK_HEADER_COLOR)]]


block_3 = [[sg.Text('Aerostat Detection', font='Any 20')],
            [sg.Button('Start Aerostat Detection')]  ]

block_4 = [[sg.Text('Camouflaged Detection', font='Any 20')],
            [sg.Button('Start Camouflage Detection')]]


layout = [[sg.Column(top_banner, size=(960, 60), pad=(0,0), background_color=DARK_HEADER_COLOR)],
          [sg.Column([[sg.Column(block_3, size=(450,320),  pad=BPAD_LEFT_INSIDE)]], pad=BPAD_LEFT, background_color=BORDER_COLOR),
           sg.Column(block_4, size=(450, 320), pad=BPAD_RIGHT)]]

window = sg.Window('Dashboard PySimpleGUI-Style', layout, margins=(0,0), default_button_element_size=(12,3), background_color=BORDER_COLOR, grab_anywhere=True)

while True:             # Event Loop
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Start Aerostat Detection':
        subprocess.call('C:/Python310/python.exe C:/Users/aloni/Documents/GitHub/Drone-Balloon-Camouflage-RealTime-Detection/YOLOV5/detect.py --source 0 --weights C:/Users/aloni/Documents/GitHub/Drone-Balloon-Camouflage-RealTime-Detection/YOLOV5/best.pt --conf-thres 0.7 --half --dnn', shell=True)

window.close()
