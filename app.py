from flask import Flask, render_template, request, flash, redirect, url_for


app = Flask(__name__)
import mysql.connector
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pprint import pprint
from time import sleep
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama,OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from flask import Flask, json
from flask_mail import Mail, Message
import numpy as np
import re
from itsdangerous import URLSafeTimedSerializer
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import random


from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo



from flask import Flask, flash, render_template, redirect, url_for
app.secret_key = 'hello'
 # Change 5001 to any available port


conn=mysql.connector.connect(host='localhost', username='root',password='Mishika@07',database= 'neaDraft')
cursor=conn.cursor()
print("Connection successful yay!")

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='1fa3b4e9cc7a4159a6d5b6d29b93938a',
                                               client_secret='f73e2a9130f1462a8cdf2e8967721352',
                                               redirect_uri='http://127.0.0.1:5001/list_page',
                                               scope='user-library-read playlist-read-private'))

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'mishika.khurana@gmail.com'
app.config['MAIL_PASSWORD'] = 'omfy mokj qszh prkn'
app.config['MAIL_DEFAULT_SENDER'] = 'mishika.khurana@gmail.com'

mail = Mail(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/list_page')
def listoutput():
    return render_template('list.html')

@app.route('/hello', methods=['GET', 'POST'])
def hello():
    userEmail = request.form.get("Email")
    userPassword = request.form.get("password")
    cursor.execute("SELECT password FROM logIn WHERE email=%s", (userEmail,))
    results = cursor.fetchone()
    
    if results == None:
        print ("Invalid")
        flash("There is no account with this email.", "error")
        return render_template('index.html')
    
    elif userPassword == results[0]:
        print("valid")
        return render_template('home.html')
    
    else:
        print ("Invalid")
        flash("Email or password incorrect", "error")
        return render_template('index.html')
        
   

    #print (request.form.get("Email"))

@app.route('/home_page')
def home_page():
    return render_template('home.html')

def generate_reset_token(email):
    serializer = URLSafeTimedSerializer('hello')
    return serializer.dumps(email, salt="password-reset-salt")

def verify_reset_token(token, expiration=3600):  # Token expires in 1 hour
    serializer = URLSafeTimedSerializer('hello')
    try:
        email = serializer.loads(token, salt="password-reset-salt", max_age=expiration)
        return email
    except:
        return None

@app.route('/forgotPassword')
def forgotPassword():
    return render_template('forgot.html')

@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
    emailToReset = request.form.get("Email")
    print (emailToReset)
    cursor.execute("SELECT email FROM logIn WHERE email=%s", (emailToReset,))
    results = cursor.fetchone()
    print("test1")
    if results is not None:
        print("test2")
        token = generate_reset_token(emailToReset)
        reset_url = url_for("reset_password", token=token, _external=True)
        msg = Message("Password Reset Request", recipients=[emailToReset])
        msg.body = f"Click the link to reset your password, please note the link expires in 1 hour: {reset_url}"
        mail.send(msg)
    
            
        flash("A password reset link has been sent to your email.", "success")
        return redirect(url_for("login"))
    else:
        print("test3")
        print("Email not found")
        flash("No account found with this email.", "error")
        return render_template('forgot.html')
        
        
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = URLSafeTimedSerializer.loads(token, max_age=3600) 
        if request.method == 'POST':
            newPass = request.form.get('password')
            newPass2 = request.form.get('password')
            if(newPass == newPass2):
                cursor.execute("UPDATE logIn SET password=%s WHERE email=%s", (newPass, email))
                conn.commit()
                flash('Your password has been updated!', 'error')
            
           
            
            return redirect(url_for('login'))  
    except Exception as e:
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('forgotPassword'))  

    return render_template('reset_password.html', token=token, email = email)  

 
    return redirect(url_for('home_page'))
    

@app.route('/addWord', methods=['GET', 'POST'])
def word_input():
    track_Ids = []
    Word1 = request.form.get("Word1")
    Word2 = request.form.get("Word2")
    Word3 = request.form.get("Word3")
    rec = suggestion(Word1, Word2, Word3)
    
    if rec is None:
        return render_template('home.html')
    else:
        for r in rec:
            print(r.metadata)
            fullId = r.metadata.get("song_id")
            track_Ids.append(fullId)

        # Fetch song details from Spotify API
    song_details = []
    if track_Ids:
        try:
            spotify_tracks = sp.tracks(track_Ids)  # Get tracks by IDs using Spotipy
            for track in spotify_tracks['tracks']:
                song_details.append({
                    'title': track['name'],
                    'artist': ", ".join([artist['name'] for artist in track['artists']]),
                    'album': track['album']['name'],
                    'release_date': track['album']['release_date'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms'],
                    'preview_url': track.get('preview_url'),
                    'url': track['external_urls']['spotify'],
                    'spotify_id': track['id']  # Add Spotify track ID here
                })
        except Exception as e:
            print(f"Error fetching Spotify data: {e}")

    # Pass data to the template
    return render_template('list.html', Word1=Word1, Word2=Word2, Word3=Word3, songs=song_details)


           



        
@app.route('/list_page')    
def list_page():
    return render_template('list.html')


@app.route('/signup_page')
def signup_page():
    return render_template('signup.html')

@app.route('/addUser', methods=['GET', 'POST'])
def signUp():
    name = request.form.get("Name")
    surname = request.form.get("Surname")
    newEmail = request.form.get("Email2")
    userPassword = request.form.get("Password")
    userPassword2 = request.form.get("Password2")
    cursor.execute("SELECT userID FROM logIn WHERE email=%s", (newEmail,))
    newResults = cursor.fetchone()
    upperChars = sum(1 for c in userPassword if c.isupper())
    lowerChars = sum(1 for c in userPassword if c.islower())
    digits = sum(1 for c in userPassword if c.isdigit())
    specialChars = sum(1 for c in userPassword if not c.isalnum())

    valid = re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', newEmail)
    length = len(userPassword)
    if ((valid is not None) and (newResults == None) and (userPassword == userPassword2) and (upperChars > 0 and lowerChars > 0 and digits > 0 and specialChars > 0 and length >= 8)):
       cursor.execute("INSERT INTO logIn (firstName, surname, email, password) VALUES(%s, %s, %s, %s)", (name, surname, newEmail, userPassword))
       conn.commit()
       return render_template('index.html')
    else:
        if(valid is None):
            flash("Invalid email", 'error')
        elif (newResults is not None):
            flash("Email already exists", "error")
        elif (userPassword != userPassword2):
            flash("Passwords do not match", "error")
        elif (upperChars == 0):
            flash("Password must contain at least one uppercase letter", "error")
        elif (lowerChars == 0):
            flash("Password must contain at least one lowercase letter", "error")
        elif (digits == 0):
            flash("Password must contain at least one digit", "error")
        elif (specialChars == 0):
            flash("Password must contain at least one special character", "error")
        elif (length < 8):
            flash("Password must be at least 8 characters long", "error")
        elif (userPassword != userPassword2):
            flash("Passwords do not match", "error")
            
        return render_template('signup.html')
    
    
def suggestion(Word1, Word2, Word3):

    model = OllamaLLM(model="llama3.1", temperature=0.0)

    embedding = OllamaEmbeddings(model="llama3.1")
    persist_directory = 'data/chroma_3words/'
    
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    query = f" Find songs related to {Word1} , {Word2} and {Word3}"
    query_embedding = embedding.embed_query(query)
    query_embedding = np.array(query_embedding)
    query_embedding /= np.linalg.norm(query_embedding)
    print("Query embedding:", query_embedding[:5])
    
    res = vectordb.max_marginal_relevance_search_by_vector(
        query_embedding, 
        k=25, fetch_k=100, 
        lambda_mult=0
    ) #fetches the top 25 results of semantically similar songs
    
    print(f"Retrieved {len(res)} results from Chroma.")
    
    bm25_retriever = BM25Retriever.from_documents(res, preprocess_func=word_tokenize) #conducts exact word search
    
    keyword_results = bm25_retriever.invoke(query)
    print(f"Retrieved {len(keyword_results)} results from BM25.")

    all_results = list({doc.metadata["song_id"]: doc for doc in res + keyword_results}.values())
    
    filtered_results = [
        song for song in all_results 
        if not song.metadata["summary"].startswith("I cannot provide")
    ]
    print(f"After filtering, {len(filtered_results)} results remain.")

    print(f" HELLOOO OUTPUT {filtered_results[0].metadata}")
    similarities = []
    for song in filtered_results:
        # Check if embedding exists in the metadata
        print(song.metadata)

        song_embedding = np.array(song.metadata["embedding"])  # Access the song's embedding
        song_embedding /= np.linalg.norm(song_embedding)  # Normalize the song embedding
        
        similarity = cosine_similarity([query_embedding], [song_embedding])[0][0]
        similarities.append((song, similarity))
        if not similarities:
            print("No songs with valid embeddings found.")
            return []
    
    # Sort by similarity (highest first)
    sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Get the top 10 most similar songs
    final_results = [song for song, _ in sorted_results[:10]]
    
    print(f"Returning {len(final_results)} most similar songs.")

    return final_results
    
    

def close_connection(exception):
    conn.close()
    print("Database connection closed.")
if __name__ == '__main__':
    app.run(debug=True)
