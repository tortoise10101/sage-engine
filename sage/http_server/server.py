# server.py
# Author: Thomas MINIER - MIT License 2017-2018
from markdown import markdown
from flask import Flask, Markup, render_template, request, abort, Response
from flask_cors import CORS
from sage.database.core.yaml_config import load_config
from sage.http_server.sparql_interface import sparql_blueprint
from sage.http_server.void_interface import void_blueprint
from sage.http_server.publish_query_interface import publish_query_blueprint
from sage.http_server.utils import secure_url, generate_sitemap
import datetime
import logging
import os
from sys import setrecursionlimit


def sage_app(config_file):
    """Build a Sage application with the given configuration file"""
    # set recursion depth (due to pyparsing issues)
    setrecursionlimit(3000)

    # Initialize the HTTP application with COTS enabled
    app = Flask(__name__)
    start_date = datetime.datetime.now()
    CORS(app)

    # get Gunicorn logger
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

    # Build RDF dataset from the configuration file
    dataset = load_config(config_file)

    @app.context_processor
    def inject_user():
        config = dict()
        if dataset.analytics is not None:
            config["google_analytics"] = dataset.analytics
        # inject current year
        config["now_year"] = datetime.datetime.now().year
        return dict(config=config)

    @app.route('/')
    def index():
        try:
            url = secure_url(request.url)
            dinfos = [dinfo for dinfo in dataset.describe(url)]
            long_description = Markup(markdown(dataset.long_description))
            return render_template("website/index.html", dataset=dinfos, server_public_url=dataset.public_url, default_query=dataset.default_query, long_description=long_description)
        except Exception as e:
            gunicorn_logger.error(e)
            abort(500)

    @app.route('/sparql11-compliance')
    def sparql11_compliance():
        return render_template('website/sparql11_compliance.html')

    @app.route('/sage-voc')
    def voc():
        return app.send_static_file('sage-voc.ttl')

    @app.route('/documentation')
    def doc():
        return render_template("website/documentation.html")

    @app.route('/api')
    def open_api():
        return app.send_static_file('api.yaml')

    @app.route('/specs')
    def specs():
        try:
            specs = {
                'load_avg': os.getloadavg(),
                'cpu_count': os.cpu_count()
            }
            return render_template("website/specs.html", specs=specs)
        except Exception as e:
            gunicorn_logger.error(e)
            abort(500)

    @app.route('/sitemap.xml')
    def sitemap():
        return Response(generate_sitemap(dataset, start_date.strftime("%Y-%m-%d")), content_type="application/xml")

    app.register_blueprint(sparql_blueprint(dataset))
    app.register_blueprint(void_blueprint(dataset, gunicorn_logger))
    app.register_blueprint(publish_query_blueprint(dataset, gunicorn_logger))
    return app
