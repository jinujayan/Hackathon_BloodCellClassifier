import datetime
from flask import Flask
import dash
from dash.dependencies import Input, Output, State
from PIL import Image
import dash_core_components as dcc
import dash_html_components as html
import plotly.figure_factory as ff
import plotly.graph_objs as go

#import torch
from Utilities import model_utility
from Utilities import preprocess_utility
server = Flask(__name__)
@server.route('/', methods=['POST'])
def index():
    return "hellow world"


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.scripts.config.serve_locally = True


app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Test Image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
    html.Div([
        dcc.Graph(id='season-bar')
    ], className='predictionbar')
])


def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents)

    ])

def draw_season_bar_chart():
    ## call sequence of tasks to execute a predict call
    model = model_utility.loadClassifierModel("checkpoint_densenet_73a.pth.tar")

    processed_image = preprocess_utility.predictImagePreprocessor("imageToSave2.jpeg",224)
    #dates = results['date']
    #points = results['points'].cumsum()
    #res_uniq = results['result'].unique()
    #agg = results.groupby(['result']).count()
    class_probs = model_utility.predictImageClass("imageToSave2.jpeg", model, gpu=False, topk=4)
    print("-------back to main--------")
    print(model.class_to_idx )
    print(class_probs)

    print(f"The lympho get is --> {[t for t in class_probs if t[1].startswith('LYMPHOCYTE')][0][0]}")
    #for class_probs in class_probs:

    results_x=["LYMPHOCYTE", "NEUTROPHIL", "EOSINOPHIL","MONOCYTE"]
    lymph_prob = [t for t in class_probs if t[1].startswith('LYMPHOCYTE')][0][0]
    Neutro_prob=[t for t in class_probs if t[1].startswith('NEUTROPHIL')][0][0]
    Eosino_prob = [t for t in class_probs if t[1].startswith('EOSINOPHIL')][0][0]
    Mono_prob = [t for t in class_probs if t[1].startswith('MONOCYTE')][0][0]
    count_y = [lymph_prob,Neutro_prob,Eosino_prob,Mono_prob]
    figure = go.Figure(
        data=[
            go.Bar(x=results_x, y=count_y)
        ],
        layout=go.Layout(
            title='Class probabilities',
            showlegend=True
        )
    )

    return figure

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('season-bar', component_property='figure'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def handle_bar (list_of_contents, list_of_names, list_of_dates):
    print("show list of contents.... and names")
    print(list_of_contents)
    print(list_of_names)
    if list_of_contents is not None:
        import base64

        new_content = bytes(list_of_contents[0].split(",")[1], 'utf-8')
        #new_content=base64.b64encode(new_content)
        print(new_content)

        fh = open("imageToSave2.jpeg", "wb")
        fh.write(base64.b64decode(new_content))
        #base64.b64decode(new_content)
        fh.close()


        #image = Image.open(list_of_names[0])
        #width, height = image.size
        #print(f"width and height is {width}, {height}")
        #image=list_of_contents[0]
        print("file saved to local")
        bar = draw_season_bar_chart()
        return bar
    else:
        return ""



if __name__ == '__main__':
    #app.run_server(debug=True)
    #app.run(host='0.0.0.0')
    #if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')