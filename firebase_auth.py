import streamlit as st
import re
import time
from datetime import datetime

# Hardcoded users for demonstration (replace with proper authentication later)
DEMO_USERS = {
    "user@example.com": "Password123",
    "admin@vhydro.com": "Admin123"
}

# Function to validate email format
def is_valid_email(email):
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(email_pattern, email))

# Function to validate password strength
def is_strong_password(password):
    # At least 8 characters, with 1 uppercase, 1 lowercase, 1 digit
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.islower() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True

# Function to handle signup
def signup(email, password, confirm_password):
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    if password != confirm_password:
        return False, "Passwords don't match"
    
    if not is_strong_password(password):
        return False, "Password must be at least 8 characters with 1 uppercase, 1 lowercase, and 1 digit"
    
    if email in DEMO_USERS:
        return False, "Email already in use"
    
    # Add user to demo users (in a real app, you would save to a database)
    DEMO_USERS[email] = password
    
    # Store user info in session state
    st.session_state["user_id"] = email
    st.session_state["email"] = email
    st.session_state["logged_in"] = True
    
    return True, "Account created successfully!"

# Function to handle login
def login(email, password):
    if not email or not password:
        return False, "Please enter both email and password"
    
    # Check if user exists and password matches
    if email in DEMO_USERS and DEMO_USERS[email] == password:
        # Store user info in session state
        st.session_state["user_id"] = email
        st.session_state["email"] = email
        st.session_state["logged_in"] = True
        
        return True, "Login successful!"
    else:
        return False, "Invalid email or password"

# Function to handle password reset (simplified mock version)
def reset_password(email):
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    if email in DEMO_USERS:
        return True, "Password reset email would be sent! (Demo only)"
    else:
        return False, "Email not found"

# Function to log out
def logout():
    for key in ["user_id", "email", "logged_in"]:
        if key in st.session_state:
            del st.session_state[key]

# Main authentication function
def authenticate():
    """
    Main authentication function.
    Returns True if user is authenticated, False otherwise.
    """
    # Check if user is already authenticated
    if st.session_state.get("logged_in", False):
        return True
    
    current_page = st.session_state.get('current_page', 'Home')
    
    # Only enforce authentication for Analysis Tool and Results Visualization pages
    if current_page in ["Analysis Tool", "Results Visualization"]:
        return False
    
    # Allow access to other pages without authentication
    return True

# Function to show user profile/account page
def user_account_page():
    st.markdown("<h2 class='sub-header'>My Account</h2>", unsafe_allow_html=True)
    
    if not st.session_state.get("logged_in", False):
        st.warning("Your session has expired. Please log in again.")
        logout()
        st.rerun()
        return
    
    # Create a card-like interface for user info
    st.markdown("""
    <style>
    .user-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-header {
        border-bottom: 1px solid #e9ecef;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .user-field {
        margin-bottom: 15px;
    }
    .field-label {
        font-weight: bold;
        color: #0e4194;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="user-card">', unsafe_allow_html=True)
    st.markdown('<div class="user-header">', unsafe_allow_html=True)
    st.markdown(f"<h3>👤 {st.session_state.get('email', 'User')}</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="user-field">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Email:</span>', unsafe_allow_html=True)
    st.write(st.session_state.get('email', 'Unknown'))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="user-field">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Account Type:</span>', unsafe_allow_html=True)
    st.write("Standard User")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="user-field">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Last Login:</span>', unsafe_allow_html=True)
    st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Account settings section
    st.markdown("<h3 class='section-header'>Account Settings</h3>", unsafe_allow_html=True)
    
    st.markdown('<div class="user-card">', unsafe_allow_html=True)
    with st.form("change_password_form"):
        st.markdown("<h4>Change Password</h4>", unsafe_allow_html=True)
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        submit_button = st.form_submit_button("Update Password")
        
        if submit_button:
            # Mock password change logic
            if current_password and new_password and confirm_password:
                if current_password == DEMO_USERS.get(st.session_state.get('email', '')):
                    if new_password == confirm_password:
                        if is_strong_password(new_password):
                            DEMO_USERS[st.session_state.get('email', '')] = new_password
                            st.success("Password updated successfully!")
                        else:
                            st.error("New password must be at least 8 characters with 1 uppercase, 1 lowercase, and 1 digit")
                    else:
                        st.error("New passwords don't match")
                else:
                    st.error("Current password is incorrect")
            else:
                st.error("Please fill all fields")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Logout section
    st.markdown("<h3 class='section-header'>Logout</h3>", unsafe_allow_html=True)
    
    st.markdown('<div class="user-card">', unsafe_allow_html=True)
    st.write("Click the button below to log out of your account.")
    
    if st.button("Logout"):
        logout()
        st.success("You have been logged out!")
        time.sleep(1)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
