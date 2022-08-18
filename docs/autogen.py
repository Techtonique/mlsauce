# -*- coding: utf-8 -*-
import pathlib
import shutil
import keras_autodoc

PAGES = {    
    'documentation/classifiers.md': [
        'mlsauce.AdaOpt',
        'mlsauce.AdaOpt.fit',
        'mlsauce.AdaOpt.predict',
        'mlsauce.AdaOpt.predict_proba',
        'mlsauce.LSBoostClassifier',
        'mlsauce.LSBoostClassifier.fit',
        'mlsauce.LSBoostClassifier.predict',
        'mlsauce.LSBoostClassifier.predict_proba',    
        'mlsauce.StumpClassifier',
        'mlsauce.StumpClassifier.fit',
        'mlsauce.StumpClassifier.predict',
        'mlsauce.StumpClassifier.predict_proba',
    ],
    'documentation/regressors.md': [
        'mlsauce.LassoRegressor',
        'mlsauce.LassoRegressor.fit',
        'mlsauce.LassoRegressor.predict',
        'mlsauce.LSBoostRegressor',
        'mlsauce.LSBoostRegressor.fit',
        'mlsauce.LSBoostRegressor.predict',
        'mlsauce.RidgeRegressor',
        'mlsauce.RidgeRegressor.fit',
        'mlsauce.RidgeRegressor.predict',
    ]
}

mlsauce_dir = pathlib.Path(__file__).resolve().parents[1]


def generate(dest_dir):
    template_dir = mlsauce_dir / 'docs' / 'templates'

    doc_generator = keras_autodoc.DocumentationGenerator(
        PAGES,
        'https://github.com/Techtonique/mlsauce/blob/master',
        template_dir,
        #mlsauce_dir / 'examples'
    )
    doc_generator.generate(dest_dir)

    readme = (mlsauce_dir / 'README.md').read_text()
    index = (template_dir / 'index.md').read_text()
    index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
    (dest_dir / 'index.md').write_text(index, encoding='utf-8')
    shutil.copyfile(mlsauce_dir / 'CONTRIBUTING.md',
                    dest_dir / 'contributing.md')
    #shutil.copyfile(mlsauce_dir / 'docs' / 'extra.css',
    #                dest_dir / 'extra.css')


if __name__ == '__main__':
    generate(mlsauce_dir / 'docs' / 'sources')