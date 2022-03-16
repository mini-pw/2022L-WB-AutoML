## Podsumowanie pracy domowej 2

<table>
<thead>
  <tr>
    <th></th>
    <th>Wykonany preprocessing</th>
    <th>CV</th>
    <th>Typ modelu ML</th>
    <th>Optymalizowane HP</th>
    <th>AUC (default)</th>
    <th>AUC (GS)</th>
    <th>AUC (RS)</th>
    <th>AUC (BO)</th>
    <th>Link do raportu</th>
  </tr>
</thead>
<tbody>
   <tr>
    <td>1</td>
    <td align='center'>---</td>
    <td align='center'>3</td>
    <td align='center'>RandomForestClassifier</td>
    <td align='center'> n_estimators, criterion, min_samples_leaf, min_samples_split</td>
    <td align='center'>0.8441008043128087</td>
    <td align='center'>0.8508267230021713</td>
    <td align='center'>0.8511484391511555</td>
    <td align='center'>0.850817036655483</td>
    <td align='center'><a href="TomaszewskiŁukasz/WB_PD_2.ipynb">link</td>
      </tr>
  <tr>
    <td>2</td>
    <td align='center'>---</td>
    <td align='center'>5</td>
    <td align='center'>RandomForestClassifier</td>
    <td align='center'> n_estimators, criterion, max_features, max_depth min_samples_split</td>
    <td align='center'>0.6600572212993829</td>
    <td align='center'>0.6515612275157766</td>
    <td align='center'>0.6137562012818499</td>
    <td align='center'>0.683000773865825</td>
    <td align='center'><a href="RoguskiMikolaj/Untitled.ipynb">link</td>
      </tr>
  <tr>
    <td>3</td>
    <td align='center'>---</td>
    <td align='center'>3</td>
    <td align='center'>RandomForestClassifier</td>
    <td align='center'> n_estimators, max_features, max_depth min_samples_split, min_samples_leaf, bootstrap</td>
    <td align='center'>0.63</td>
    <td align='center'>0.57</td>
    <td align='center'>0.58</td>
    <td align='center'>0.62</td>
    <td align='center'><a href="KruszewskiJan/hw2.html">link</td>
      </tr>
  <tr>
    <td>4</td>
    <td align='center'>---</td>
    <td align='center'>3</td>
    <td align='center'>RandomForestClassifier</td>
    <td align='center'> n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, bootstrap, criterion</td>
    <td align='center'>0.6756</td>
    <td align='center'>0.6919</td>
    <td align='center'>0.6919</td>
    <td align='center'>0.6887</td>
    <td align='center'><a href="https://github.com/MI2-Education/2022L-WB-AutoML/tree/main/homeworks/hw2/Grzegorz_Zbrze%C5%BCny">link</td>
      </tr>
  <tr>
    <td>5</td>
    <td align='center'>---</td>
    <td align='center'>3</td>
    <td align='center'>XGBClassifier</td>
    <td align='center'> gamma, learning_rate, max_depth, n_estimators, reg_alpha, reg_lambda</td>
    <td align='center'>0.6913</td>
    <td align='center'>0.6940</td>
    <td align='center'>0.6963</td>
    <td align='center'>0.6957</td>
    <td align='center'><a href="GałkowskiMikołaj/hw2_Gałkowski">link</td>
      </tr> 
      <tr>
    <td>6</td>
    <td align='center'>---</td>
    <td align='center'>5</td>
    <td align='center'>XGBClassifier</td>
    <td align='center'>gamma, learning_rate, max_depth, min_child_weight, subsample</td>
    <td align='center'>0.6910</td>
    <td align='center'>0.6936</td>
    <td align='center'>0.6953</td>
    <td align='center'>0.6955</td>
    <td align='center'><a href="MarciniakPiotr/homework.ipynb">link</td>
      </tr>
      <tr>
    <td>7</td>
    <td align='center'>---</td>
    <td align='center'>3</td>
    <td align='center'>LGBMClassifier</td>
    <td align='center'>max_depth, min_data_in_leaf, n_estimators, num_leaves</td>
    <td align='center'>0.7287</td>
    <td align='center'>-</td>
    <td align='center'>0.7295</td>
    <td align='center'>0.7283</td>
    <td align='center'><a href="KomorowskiMichal/homework2.ipynb">link</td>
      </tr> 
    <tr>
    <td>8</td>
    <td align='center'>---</td>
    <td align='center'>5</td>
    <td align='center'>GradBoostClassifier</td>
    <td align='center'>criterion, learning_rate, max_depth</td>
    <td align='center'>0.8691564973776389</td>
    <td align='center'>0.8744405506515862</td>
    <td align='center'>0.87320370593767</td>
    <td align='center'>0.8743865410403371</td>
    <td align='center'><a href="KurowskiKacper/[WB2]_PD2_Kacper_Kurowski.ipynb">link</td>
      </tr>  
  </tbody>
</table>

Legenda:
- CV - ile podziałów rozważano w kroswalidcji
- HP - hiperparametry
- AUC (default) - AUC modelu z domyślnymi hiperparametrami
- AUC (GS) - AUC modelu zoptymalizownego metodą GridSearch
- AUC (RS) - AUC modelu zoptymalizownego metodą RandomSearch
- AUC (BO) - AUC modelu zoptymalizownego optymalizacją Bayesowską
